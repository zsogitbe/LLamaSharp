using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using LLama.Exceptions;
using LLama.Native;

namespace LLama.Speculative
{
    /// <summary>
    /// Contains the verification results for a single sequence after a speculative decode pass.
    /// </summary>
    public readonly struct LlamaSpeculativeResult
    {
        /// <summary>
        /// The native sequence identifier (<c>llama_seq_id</c>) these results map to.
        /// </summary>
        public LLamaSeqId SequenceId { get; }

        /// <summary>
        /// The total number of tokens successfully evaluated and accepted during this burst.
        /// <para><b>Important:</b> Includes both the accepted draft tokens and the 1 base/correction target token sampled at divergence (e.g., 3 accepted drafts returns 4).</para>
        /// </summary>
        public int Count { get; }

        /// <summary>
        /// Accepted token IDs. Only the first <see cref="Count"/> elements are valid.
        /// </summary>
        public LLamaToken[] Tokens { get; }

        /// <summary>
        /// Creates a new speculative verification result.
        /// </summary>
        /// <param name="sequenceId">The sequence identifier.</param>
        /// <param name="tokens">The accepted tokens buffer.</param>
        /// <param name="count">Number of valid tokens in <paramref name="tokens"/>.</param>
        public LlamaSpeculativeResult(LLamaSeqId sequenceId, LLamaToken[] tokens, int count)
        {
            SequenceId = sequenceId;
            Tokens = tokens;
            Count = count;
        }
    }

    /// <summary>
    /// Managed speculative decoding engine built on the official <c>llama.cpp</c> v1.0 C API.
    /// <para>Orchestrates Draft-Simple (dual-model) and Multi-Token Prediction (MTP / NextN) burst drafting, target verification, KV-cache synchronization, and hybrid/RNN-safe state rollbacks.</para>
    /// </summary>
    public sealed class SpeculativeDecoder : IDisposable
    {
        private const uint StateSeqFlagPartialOnly = 1;
        private const uint StateSeqFlagOnDevice = 2;

        private readonly SafeLLamaContextHandle _targetContext;
        private readonly SafeLLamaContextHandle _draftContext;
        private readonly SafeLLamaSamplerChainHandle _targetSampler;
        private readonly SafeLLamaSamplerChainHandle _draftSampler;
        private readonly SafeLLamaBatchExtHandle? _draftBatchExt;

        private readonly int _draftTokens;
        private readonly bool _useMtp;
        private readonly int _nLayerNextn;
        private readonly int _nEmbdOut;
        private readonly int _targetBatchSize;
        private readonly int _targetCtxSeq;
        private readonly bool _useDeviceStateRows;
        private readonly bool _useDraftDeviceStateRows;
        private readonly bool _targetNeedsStateRollback;
        private readonly bool _draftNeedsStateRollback;

        // Per-sequence pending target hidden state h_nextn from the last evaluated target token (used in MTP mode).
        private readonly Dictionary<int, float[]> _pendingH = new();
        private readonly float[] _scratchH;
        private readonly LLamaBatch _verifyBatch = new();
        private readonly LLamaBatch _draftBatch = new();
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="SpeculativeDecoder"/> class.
        /// </summary>
        /// <param name="targetContext">Safe handle to the primary target model context used for verification.</param>
        /// <param name="draftContext">Safe handle to the draft context. In MTP mode, this must be a distinct <see cref="LLamaContextType.Mtp"/> context paired with <paramref name="targetContext"/> via <c>CtxOther</c>.</param>
        /// <param name="draftTokens">Maximum number of draft tokens to propose per speculative burst.</param>
        /// <param name="useMtp"><c>true</c> to use Multi-Token Prediction (NextN hidden-state routing); <c>false</c> for standard autoregressive dual-model drafting.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="draftTokens"/> is not positive.</exception>
        /// <exception cref="RuntimeError">Thrown if handles are invalid or MTP requirements are not satisfied.</exception>
        public SpeculativeDecoder(
            SafeLLamaContextHandle targetContext,
            SafeLLamaContextHandle draftContext,
            int draftTokens = 16,
            bool useMtp = false)
        {
            if (draftTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(draftTokens), "Draft token budget must be greater than zero.");

            if (targetContext is null || targetContext.IsInvalid || targetContext.IsClosed ||
                draftContext is null || draftContext.IsInvalid || draftContext.IsClosed)
            {
                throw new RuntimeError("Cannot initialize SpeculativeDecoder: Target or draft context handle is invalid.");
            }

            IntPtr targetPtr = targetContext.DangerousGetHandle();
            IntPtr draftPtr = draftContext.DangerousGetHandle();

            if (useMtp && targetPtr == draftPtr)
            {
                throw new RuntimeError(
                    "In llama.cpp v1.0, MTP speculative decoding requires a dedicated draft context initialized with ContextType = LLamaContextType.Mtp and CtxOther set to the target context handle.");
            }

            _targetContext = targetContext;
            _draftContext = draftContext;
            _draftTokens = draftTokens;
            _useMtp = useMtp;

            IntPtr targetModel = NativeMethods.llama_get_model(targetPtr);
            IntPtr draftModel = NativeMethods.llama_get_model(draftPtr);

            _nEmbdOut = NativeMethods.llama_model_n_embd_out(targetModel);
            _targetBatchSize = checked((int)NativeMethods.llama_n_batch(targetPtr));
            _targetCtxSeq = checked((int)NativeMethods.llama_n_ctx_seq(targetPtr));

            _targetNeedsStateRollback =
                (NativeMethods.llama_model_is_recurrent(targetModel) || NativeMethods.llama_model_is_hybrid(targetModel)) &&
                NativeApi.llama_n_rs_seq(targetPtr) == 0;

            _draftNeedsStateRollback =
                !useMtp &&
                (NativeMethods.llama_model_is_recurrent(draftModel) || NativeMethods.llama_model_is_hybrid(draftModel)) &&
                NativeApi.llama_n_rs_seq(draftPtr) == 0;

            _targetSampler = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());
            _targetSampler.AddGreedySampler();

            _draftSampler = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());
            _draftSampler.AddGreedySampler();

            try
            {
                if (_useMtp)
                {
                    _nLayerNextn = Math.Max(
                        NativeApi.llama_model_n_layer_nextn(draftModel),
                        NativeApi.llama_model_n_layer_nextn(targetModel));
                    if (_nLayerNextn <= 0)
                    {
                        throw new RuntimeError(
                            "Target model reports 0 NextN/MTP layers. Ensure the GGUF contains MTP weights and ModelParams.LoadMTP is set to true.");
                    }

                    if (_nEmbdOut <= 0)
                        throw new RuntimeError("Target model reported invalid output embedding dimension for MTP.");

                    _scratchH = new float[_nEmbdOut];

                    // Capture unmasked NextN hidden states for all decoded target tokens.
                    NativeApi.llama_set_embeddings_nextn(targetPtr, value: true, masked: false);

                    // Prefer zero-copy device buffer routing when supported by the backend.
                    bool devRows = NativeApi.llama_set_embeddings_nextn_dev(targetPtr, true)
                                && NativeApi.llama_set_ctx_state_src(draftPtr, targetPtr);
                    if (!devRows)
                    {
                        NativeApi.llama_set_ctx_state_src(draftPtr, IntPtr.Zero);
                        NativeApi.llama_set_embeddings_nextn_dev(targetPtr, false);
                    }
                    _useDeviceStateRows = devRows;

                    // Enable NextN hidden states on the draft context when chaining > 1 draft step.
                    if (_draftTokens > 1)
                    {
                        NativeApi.llama_set_embeddings_nextn(draftPtr, value: true, masked: false);
                        _useDraftDeviceStateRows = _useDeviceStateRows && NativeApi.llama_set_embeddings_nextn_dev(draftPtr, true);
                    }

                    _draftBatchExt = SafeLLamaBatchExtHandle.Create(_draftContext);
                }
                else
                {
                    _scratchH = Array.Empty<float>();
                }
            }
            catch
            {
                _draftBatchExt?.Dispose();
                _draftSampler.Dispose();
                _targetSampler.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Synchronizes MTP hidden-state tracking and warms the MTP draft context immediately after a prompt batch has been decoded on the target context.
        /// <para>No-op when not running in MTP mode.</para>
        /// </summary>
        /// <param name="promptBatch">The most recently decoded target prompt batch.</param>
        public void ProcessPrompt(LLamaBatch promptBatch)
        {
            AssertNotDisposed();
            if (!_useMtp || promptBatch is null || promptBatch.TokenCount == 0)
                return;

            var entries = ExtractBatchEntries(promptBatch);
            if (entries.Count == 0)
                return;

            IntPtr targetPtr = _targetContext.DangerousGetHandle();
            IntPtr draftPtr = _draftContext.DangerousGetHandle();
            IntPtr draftExtPtr = _draftBatchExt!.DangerousGetHandle();

            // Group token entries by sequence while preserving original target batch row indices.
            var perSeq = new Dictionary<int, List<(int BatchRow, int Token, int Pos)>>();
            for (int row = 0; row < entries.Count; row++)
            {
                var e = entries[row];
                if (!perSeq.TryGetValue(e.SeqId, out var list))
                {
                    list = new List<(int, int, int)>();
                    perSeq[e.SeqId] = list;
                }
                list.Add((row, e.Token, e.Pos));
            }

            NativeApi.llama_set_nextn_layer_offset(draftPtr, 0);
            if (_useDeviceStateRows)
                NativeApi.llama_set_ctx_state_src(draftPtr, targetPtr);

            // Feed (token[k], pos[k-1], h_nextn[row[k-1]]) into the MTP context to warm its KV cache.
            foreach (var kvp in perSeq)
            {
                int seqId = kvp.Key;
                var seqTokens = kvp.Value;

                bool hasPriorH = _pendingH.TryGetValue(seqId, out var priorH);
                int startIndex = hasPriorH ? 0 : 1;

                if (startIndex < seqTokens.Count)
                {
                    _draftBatchExt.Clear();

                    if (_useDeviceStateRows)
                    {
                        int scratchRow = _targetBatchSize + seqId;
                        if (hasPriorH)
                            UploadDeviceNextnRow(targetPtr, scratchRow, priorH!);

                        for (int k = startIndex; k < seqTokens.Count; k++)
                        {
                            int stateRow = (k == 0) ? scratchRow : seqTokens[k - 1].BatchRow;
                            int pos = (k == 0) ? seqTokens[0].Pos - 1 : seqTokens[k - 1].Pos;
                            if (pos < 0)
                                continue;

                            int idx = NativeApi.llama_batch_ext_add_token(draftExtPtr, seqId, seqTokens[k].Token);
                            if (idx < 0)
                                throw new RuntimeError($"llama_batch_ext_add_token failed with code {idx}.");

                            NativeApi.llama_batch_ext_set_embd_state_row(draftExtPtr, idx, stateRow);
                            NativeApi.llama_batch_ext_set_pos(draftExtPtr, idx, ref pos);
                            NativeApi.llama_batch_ext_set_output_logits(draftExtPtr, idx, false);
                        }

                        int rc = NativeApi.llama_process(draftPtr, LLamaProcessType.Decode, draftExtPtr);
                        if (rc != 0)
                            throw new LLamaDecodeError((DecodeResult)rc);
                    }
                    else
                    {
                        // Host path: pin priorH if needed; target host embeddings remain valid until next target decode.
                        var priorHandle = hasPriorH ? GCHandle.Alloc(priorH, GCHandleType.Pinned) : default;
                        try
                        {
                            for (int k = startIndex; k < seqTokens.Count; k++)
                            {
                                int pos = (k == 0) ? seqTokens[0].Pos - 1 : seqTokens[k - 1].Pos;
                                if (pos < 0)
                                    continue;

                                IntPtr statePtr = (k == 0)
                                    ? priorHandle.AddrOfPinnedObject()
                                    : NativeApi.llama_get_embeddings_nextn_ith(targetPtr, seqTokens[k - 1].BatchRow);

                                if (statePtr == IntPtr.Zero)
                                    continue;

                                int idx = NativeApi.llama_batch_ext_add_token(draftExtPtr, seqId, seqTokens[k].Token);
                                if (idx < 0)
                                    throw new RuntimeError($"llama_batch_ext_add_token failed with code {idx}.");

                                var embd = new LLamaEmbd { data = statePtr, n_rows = 1, n_embd = (nuint)_nEmbdOut };
                                NativeApi.llama_batch_ext_set_embd_state(draftExtPtr, idx, embd);
                                NativeApi.llama_batch_ext_set_pos(draftExtPtr, idx, ref pos);
                                NativeApi.llama_batch_ext_set_output_logits(draftExtPtr, idx, false);
                            }

                            int rc = NativeApi.llama_process(draftPtr, LLamaProcessType.Decode, draftExtPtr);
                            if (rc != 0)
                                throw new LLamaDecodeError((DecodeResult)rc);
                        }
                        finally
                        {
                            if (priorHandle.IsAllocated)
                                priorHandle.Free();
                        }
                    }
                }

                // Store the final token's h_nextn so the next generated base token can draft immediately.
                int lastRow = seqTokens[seqTokens.Count - 1].BatchRow;
                CaptureTargetNextnRow(targetPtr, lastRow, seqId);
            }
        }

        /// <summary>
        /// Clears any cached per-sequence MTP hidden states when a sequence is disposed or modified.
        /// </summary>
        /// <param name="seqId">The sequence identifier to clear.</param>
        public void RemoveSequence(LLamaSeqId seqId)
        {
            _pendingH.Remove((int)seqId);
        }

        /// <summary>
        /// Clones cached MTP hidden-state tracking when a conversation sequence is forked.
        /// </summary>
        /// <param name="srcSeqId">The source sequence identifier.</param>
        /// <param name="dstSeqId">The newly forked sequence identifier.</param>
        public void ForkSequence(LLamaSeqId srcSeqId, LLamaSeqId dstSeqId)
        {
            if (_pendingH.TryGetValue((int)srcSeqId, out var srcH))
            {
                var copy = new float[srcH.Length];
                Array.Copy(srcH, copy, srcH.Length);
                _pendingH[(int)dstSeqId] = copy;
            }
        }

        /// <summary>
        /// Executes a complete speculative decoding pipeline (Draft, Verify, and Rollback) for the provided batch.
        /// <para>Supports single- and multi-sequence generation batches where each active sequence contributes its next base token.</para>
        /// </summary>
        /// <param name="batch">Batch containing the next base token(s) to evaluate and speculate from.</param>
        /// <param name="maxSequences">Maximum number of sequence results to return. Defaults to 128.</param>
        /// <returns>An array of <see cref="LlamaSpeculativeResult"/> containing the accepted tokens for each active sequence.</returns>
        public LlamaSpeculativeResult[] Decode(LLamaBatch batch, int maxSequences = 128)
        {
            AssertNotDisposed();

            if (batch is null || batch.TokenCount == 0)
                return Array.Empty<LlamaSpeculativeResult>();

            var entries = ExtractBatchEntries(batch);
            if (entries.Count == 0)
                return Array.Empty<LlamaSpeculativeResult>();

            int count = Math.Min(entries.Count, maxSequences);
            var results = new LlamaSpeculativeResult[count];

            for (int i = 0; i < count; i++)
            {
                var entry = entries[i];
                results[i] = _useMtp
                    ? DecodeSequenceMtp(entry.SeqId, (LLamaToken)entry.Token, entry.Pos)
                    : DecodeSequenceDraftSimple(entry.SeqId, (LLamaToken)entry.Token, entry.Pos);
            }

            return results;
        }

        private LlamaSpeculativeResult DecodeSequenceDraftSimple(int seqId, LLamaToken baseToken, int basePos)
        {
            IntPtr targetPtr = _targetContext.DangerousGetHandle();
            IntPtr draftPtr = _draftContext.DangerousGetHandle();

            int budget = Math.Min(_draftTokens, Math.Max(0, _targetCtxSeq - basePos - 1));
            if (budget == 0)
                return DecodeSingleTokenFallback(targetPtr, seqId, baseToken, basePos, syncDraft: true);

            // Snapshot recurrent/hybrid states if automatic n_rs_seq rollback is not enabled.
            uint targetSnapFlags = 0;
            byte[]? targetSnapshot = _targetNeedsStateRollback ? CapturePartialState(targetPtr, seqId, out targetSnapFlags) : null;
            uint draftSnapFlags = 0;
            byte[]? draftSnapshot = _draftNeedsStateRollback ? CapturePartialState(draftPtr, seqId, out draftSnapFlags) : null;

            // 1. Autoregressive drafting on ctx_dft starting from (baseToken @ basePos).
            var draftProposed = new LLamaToken[budget];
            LLamaToken current = baseToken;
            for (int k = 0; k < budget; k++)
            {
                _draftBatch.Clear();
                _draftBatch.Add(current, basePos + k, (LLamaSeqId)seqId, logits: true);
                DecodeBatchOrThrow(draftPtr, _draftBatch);

                current = _draftSampler.Sample(_draftContext, 0);
                draftProposed[k] = current;
            }

            // 2. Verify [baseToken @ basePos, draftProposed[0..budget-1] @ basePos+1..basePos+budget] in one target pass.
            _verifyBatch.Clear();
            _verifyBatch.Add(baseToken, basePos, (LLamaSeqId)seqId, logits: true);
            for (int k = 0; k < budget; k++)
                _verifyBatch.Add(draftProposed[k], basePos + 1 + k, (LLamaSeqId)seqId, logits: true);

            DecodeBatchOrThrow(targetPtr, _verifyBatch);

            // 3. Greedy verification against target logits.
            var accepted = new LLamaToken[budget + 1];
            int acceptedDrafts = 0;
            for (int k = 0; k <= budget; k++)
            {
                LLamaToken targetToken = _targetSampler.Sample(_targetContext, k);
                accepted[k] = targetToken;

                if (k < budget && targetToken == draftProposed[k])
                {
                    acceptedDrafts++;
                }
                else
                {
                    break;
                }
            }

            int totalCount = acceptedDrafts + 1;

            // 4. Synchronize target KV cache: retain positions [0 .. basePos + acceptedDrafts].
            int targetKeepEnd = basePos + acceptedDrafts + 1;
            if (acceptedDrafts < budget)
            {
                RollbackSequence(
                    targetPtr,
                    seqId,
                    targetKeepEnd,
                    targetSnapshot,
                    targetSnapFlags,
                    baseToken,
                    basePos,
                    draftProposed,
                    acceptedDrafts);
            }

            // 5. Synchronize draft KV cache to the exact same position [0 .. basePos + acceptedDrafts].
            if (acceptedDrafts == budget)
            {
                // Draft evaluated up to basePos + budget - 1; feed the last accepted draft token at basePos + budget.
                _draftBatch.Clear();
                _draftBatch.Add(draftProposed[budget - 1], basePos + budget, (LLamaSeqId)seqId, logits: false);
                DecodeBatchOrThrow(draftPtr, _draftBatch);
            }
            else
            {
                // Draft evaluated up to basePos + budget - 1; purge rejected suffix [basePos + acceptedDrafts + 1, inf).
                RollbackSequence(
                    draftPtr,
                    seqId,
                    targetKeepEnd,
                    draftSnapshot,
                    draftSnapFlags,
                    baseToken,
                    basePos,
                    draftProposed,
                    acceptedDrafts);
            }

            return new LlamaSpeculativeResult((LLamaSeqId)seqId, accepted, totalCount);
        }

        private LlamaSpeculativeResult DecodeSequenceMtp(int seqId, LLamaToken baseToken, int basePos)
        {
            IntPtr targetPtr = _targetContext.DangerousGetHandle();
            IntPtr draftPtr = _draftContext.DangerousGetHandle();
            IntPtr draftExtPtr = _draftBatchExt!.DangerousGetHandle();

            int budget = Math.Min(_draftTokens, Math.Max(0, _targetCtxSeq - basePos - 1));

            // Cold start or context boundary: decode baseToken normally, capture h_nextn, and return 1 token.
            if (budget == 0 || basePos <= 0 || !_pendingH.TryGetValue(seqId, out var pendingState))
            {
                var fallback = DecodeSingleTokenFallback(targetPtr, seqId, baseToken, basePos, syncDraft: false);
                CaptureTargetNextnRow(targetPtr, 0, seqId);
                return fallback;
            }

            uint targetSnapFlags = 0;
            byte[]? targetSnapshot = _targetNeedsStateRollback ? CapturePartialState(targetPtr, seqId, out targetSnapFlags) : null;

            // 1. Draft `budget` tokens via MTP head(s) using pendingState (from basePos - 1) and chained draft states.
            var draftProposed = new LLamaToken[budget];
            int draftBasePos = basePos - 1;

            try
            {
                for (int k = 0; k < budget; k++)
                {
                    int headOffset = Math.Min(k, _nLayerNextn - 1);
                    NativeApi.llama_set_nextn_layer_offset(draftPtr, headOffset);

                    LLamaToken inputToken = (k == 0) ? baseToken : draftProposed[k - 1];
                    int pos = draftBasePos + k;

                    _draftBatchExt.Clear();
                    int idx = NativeApi.llama_batch_ext_add_token(draftExtPtr, seqId, (int)inputToken);
                    if (idx < 0)
                        throw new RuntimeError($"llama_batch_ext_add_token failed with code {idx}.");

                    NativeApi.llama_batch_ext_set_pos(draftExtPtr, idx, ref pos);
                    NativeApi.llama_batch_ext_set_output_logits(draftExtPtr, idx, true);

                    if (k == 0)
                    {
                        if (_useDeviceStateRows)
                        {
                            int scratchRow = _targetBatchSize + seqId;
                            NativeApi.llama_set_ctx_state_src(draftPtr, targetPtr);
                            UploadDeviceNextnRow(targetPtr, scratchRow, pendingState);
                            NativeApi.llama_batch_ext_set_embd_state_row(draftExtPtr, idx, scratchRow);
                            ProcessExtOrThrow(draftPtr, draftExtPtr);
                        }
                        else
                        {
                            ProcessExtWithHostStateOrThrow(draftPtr, draftExtPtr, idx, pendingState);
                        }
                    }
                    else
                    {
                        if (_useDraftDeviceStateRows)
                        {
                            NativeApi.llama_set_ctx_state_src(draftPtr, draftPtr);
                            NativeApi.llama_batch_ext_set_embd_state_row(draftExtPtr, idx, 0);
                            ProcessExtOrThrow(draftPtr, draftExtPtr);
                        }
                        else
                        {
                            // Read previous step's host h_nextn from ctx_dft into scratch buffer before next decode overwrites it.
                            IntPtr prevDraftH = NativeApi.llama_get_embeddings_nextn_ith(draftPtr, 0);
                            if (prevDraftH == IntPtr.Zero)
                            {
                                budget = k;
                                break;
                            }
                            Marshal.Copy(prevDraftH, _scratchH, 0, _nEmbdOut);
                            ProcessExtWithHostStateOrThrow(draftPtr, draftExtPtr, idx, _scratchH);
                        }
                    }

                    draftProposed[k] = _draftSampler.Sample(_draftContext, 0);
                }
            }
            finally
            {
                NativeApi.llama_set_nextn_layer_offset(draftPtr, 0);
                if (_useDeviceStateRows)
                    NativeApi.llama_set_ctx_state_src(draftPtr, targetPtr);
            }

            // 2. Verify [baseToken @ basePos, draftProposed[0..budget-1] @ basePos+1..basePos+budget] on target context.
            _verifyBatch.Clear();
            _verifyBatch.Add(baseToken, basePos, (LLamaSeqId)seqId, logits: true);
            for (int k = 0; k < budget; k++)
                _verifyBatch.Add(draftProposed[k], basePos + 1 + k, (LLamaSeqId)seqId, logits: true);

            DecodeBatchOrThrow(targetPtr, _verifyBatch);

            // 3. Greedy verification against target logits.
            var accepted = new LLamaToken[budget + 1];
            int acceptedDrafts = 0;
            for (int k = 0; k <= budget; k++)
            {
                LLamaToken targetToken = _targetSampler.Sample(_targetContext, k);
                accepted[k] = targetToken;

                if (k < budget && targetToken == draftProposed[k])
                {
                    acceptedDrafts++;
                }
                else
                {
                    break;
                }
            }

            int totalCount = acceptedDrafts + 1;

            // 4. Capture the true target h_nextn at the last kept target batch row (= acceptedDrafts) for the next burst.
            CaptureTargetNextnRow(targetPtr, acceptedDrafts, seqId);

            // 5. Sync MTP draft KV cache: purge speculative entries at [basePos, inf) and replay accepted drafts using true target h_nextn rows [0 .. acceptedDrafts - 1].
            IntPtr draftMem = NativeMethods.llama_get_memory(draftPtr);
            NativeMethods.llama_memory_seq_rm(draftMem, seqId, basePos, -1);

            if (acceptedDrafts > 0)
            {
                _draftBatchExt.Clear();
                if (_useDeviceStateRows)
                {
                    NativeApi.llama_set_ctx_state_src(draftPtr, targetPtr);
                    for (int k = 0; k < acceptedDrafts; k++)
                    {
                        int pos = basePos + k;
                        int idx = NativeApi.llama_batch_ext_add_token(draftExtPtr, seqId, (int)draftProposed[k]);
                        if (idx < 0)
                            throw new RuntimeError($"llama_batch_ext_add_token failed with code {idx}.");

                        NativeApi.llama_batch_ext_set_embd_state_row(draftExtPtr, idx, k);
                        NativeApi.llama_batch_ext_set_pos(draftExtPtr, idx, ref pos);
                        NativeApi.llama_batch_ext_set_output_logits(draftExtPtr, idx, false);
                    }
                    ProcessExtOrThrow(draftPtr, draftExtPtr);
                }
                else
                {
                    for (int k = 0; k < acceptedDrafts; k++)
                    {
                        IntPtr trueStatePtr = NativeApi.llama_get_embeddings_nextn_ith(targetPtr, k);
                        if (trueStatePtr == IntPtr.Zero)
                            continue;

                        int pos = basePos + k;
                        int idx = NativeApi.llama_batch_ext_add_token(draftExtPtr, seqId, (int)draftProposed[k]);
                        if (idx < 0)
                            throw new RuntimeError($"llama_batch_ext_add_token failed with code {idx}.");

                        var embd = new LLamaEmbd { data = trueStatePtr, n_rows = 1, n_embd = (nuint)_nEmbdOut };
                        NativeApi.llama_batch_ext_set_embd_state(draftExtPtr, idx, embd);
                        NativeApi.llama_batch_ext_set_pos(draftExtPtr, idx, ref pos);
                        NativeApi.llama_batch_ext_set_output_logits(draftExtPtr, idx, false);
                    }
                    ProcessExtOrThrow(draftPtr, draftExtPtr);
                }
            }

            // 6. Roll back rejected target positions [basePos + acceptedDrafts + 1, inf).
            if (acceptedDrafts < budget)
            {
                RollbackSequence(
                    targetPtr,
                    seqId,
                    basePos + acceptedDrafts + 1,
                    targetSnapshot,
                    targetSnapFlags,
                    baseToken,
                    basePos,
                    draftProposed,
                    acceptedDrafts);
            }

            return new LlamaSpeculativeResult((LLamaSeqId)seqId, accepted, totalCount);
        }

        private LlamaSpeculativeResult DecodeSingleTokenFallback(
            IntPtr targetPtr,
            int seqId,
            LLamaToken baseToken,
            int basePos,
            bool syncDraft)
        {
            _verifyBatch.Clear();
            _verifyBatch.Add(baseToken, basePos, (LLamaSeqId)seqId, logits: true);
            DecodeBatchOrThrow(targetPtr, _verifyBatch);

            if (syncDraft)
            {
                _draftBatch.Clear();
                _draftBatch.Add(baseToken, basePos, (LLamaSeqId)seqId, logits: false);
                DecodeBatchOrThrow(_draftContext.DangerousGetHandle(), _draftBatch);
            }

            LLamaToken sampled = _targetSampler.Sample(_targetContext, 0);
            return new LlamaSpeculativeResult((LLamaSeqId)seqId, new[] { sampled }, 1);
        }

        private void RollbackSequence(
            IntPtr ctxPtr,
            int seqId,
            int removeFromPos,
            byte[]? preVerifySnapshot,
            uint snapshotFlags,
            LLamaToken baseToken,
            int basePos,
            LLamaToken[] draftProposed,
            int acceptedDrafts)
        {
            IntPtr mem = NativeMethods.llama_get_memory(ctxPtr);
            if (NativeMethods.llama_memory_seq_rm(mem, seqId, removeFromPos, -1))
                return;

            // Partial KV removal failed (typical on recurrent/hybrid architectures when n_rs_seq == 0).
            // Restore pre-verify state snapshot and replay only the accepted prefix to prevent positional collisions.
            if (preVerifySnapshot != null && preVerifySnapshot.Length > 0)
            {
                RestorePartialState(ctxPtr, seqId, preVerifySnapshot, snapshotFlags);
                NativeMethods.llama_memory_seq_rm(mem, seqId, basePos, -1);

                var replay = new LLamaBatch();
                replay.Add(baseToken, basePos, (LLamaSeqId)seqId, logits: false);
                for (int k = 0; k < acceptedDrafts; k++)
                    replay.Add(draftProposed[k], basePos + 1 + k, (LLamaSeqId)seqId, logits: false);

                DecodeBatchOrThrow(ctxPtr, replay);
                return;
            }

            throw new RuntimeError(
                $"Failed to roll back sequence {seqId} at position {removeFromPos}. " +
                "For recurrent or hybrid models, set RecurrentRollbackSnapshots (n_rs_seq) > 0 in context parameters.");
        }

        private void CaptureTargetNextnRow(IntPtr targetPtr, int row, int seqId)
        {
            if (!_pendingH.TryGetValue(seqId, out var buffer))
            {
                buffer = new float[_nEmbdOut];
                _pendingH[seqId] = buffer;
            }

            if (_useDeviceStateRows)
            {
                var pin = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    if (!NativeApi.llama_get_embeddings_nextn_rows(targetPtr, row, 1, pin.AddrOfPinnedObject()))
                        throw new RuntimeError($"Failed to copy h_nextn row {row} from target device buffer.");
                }
                finally
                {
                    pin.Free();
                }
            }
            else
            {
                IntPtr hostPtr = NativeApi.llama_get_embeddings_nextn_ith(targetPtr, row);
                if (hostPtr == IntPtr.Zero)
                    throw new RuntimeError($"Failed to retrieve host h_nextn embedding for row {row}.");
                Marshal.Copy(hostPtr, buffer, 0, _nEmbdOut);
            }
        }

        private void UploadDeviceNextnRow(IntPtr targetPtr, int row, float[] data)
        {
            var pin = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                if (!NativeApi.llama_set_embeddings_nextn_rows(targetPtr, row, 1, pin.AddrOfPinnedObject()))
                    throw new RuntimeError($"Failed to upload h_nextn state to target device row {row}.");
            }
            finally
            {
                pin.Free();
            }
        }

        private void ProcessExtWithHostStateOrThrow(IntPtr draftPtr, IntPtr draftExtPtr, int idx, float[] stateData)
        {
            var pin = GCHandle.Alloc(stateData, GCHandleType.Pinned);
            try
            {
                var embd = new LLamaEmbd
                {
                    data = pin.AddrOfPinnedObject(),
                    n_rows = 1,
                    n_embd = (nuint)_nEmbdOut
                };
                if (!NativeApi.llama_batch_ext_set_embd_state(draftExtPtr, idx, embd))
                    throw new RuntimeError("MTP draft context rejected host state embedding.");

                ProcessExtOrThrow(draftPtr, draftExtPtr);
            }
            finally
            {
                pin.Free();
            }
        }

        private static void ProcessExtOrThrow(IntPtr ctxPtr, IntPtr batchExtPtr)
        {
            int rc = NativeApi.llama_process(ctxPtr, LLamaProcessType.Decode, batchExtPtr);
            if (rc != 0)
                throw new LLamaDecodeError((DecodeResult)rc);
        }

        private static void DecodeBatchOrThrow(IntPtr ctxPtr, LLamaBatch batch)
        {
            using var pin = batch.ToNativeBatch(out var nativeBatch);
            int rc = NativeMethods.llama_decode(ctxPtr, nativeBatch);
            if (rc != 0)
                throw new LLamaDecodeError((DecodeResult)rc);
        }

        private static byte[]? CapturePartialState(IntPtr ctxPtr, int seqId, out uint usedFlags)
        {
            // Prefer on-device partial snapshot for speed; fall back to host partial snapshot.
            usedFlags = StateSeqFlagPartialOnly | StateSeqFlagOnDevice;
            nuint size = NativeMethods.llama_state_seq_get_size_ext(ctxPtr, seqId, usedFlags);
            if (size == 0)
            {
                usedFlags = StateSeqFlagPartialOnly;
                size = NativeMethods.llama_state_seq_get_size_ext(ctxPtr, seqId, usedFlags);
            }
            if (size == 0)
                return null;

            var buffer = new byte[checked((int)size)];
            var pin = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                nuint copied = NativeMethods.llama_state_seq_get_data_ext(ctxPtr, pin.AddrOfPinnedObject(), size, seqId, usedFlags);
                return copied > 0 ? buffer : null;
            }
            finally
            {
                pin.Free();
            }
        }

        private static void RestorePartialState(IntPtr ctxPtr, int seqId, byte[] snapshot, uint flags)
        {
            var pin = GCHandle.Alloc(snapshot, GCHandleType.Pinned);
            try
            {
                nuint restored = NativeMethods.llama_state_seq_set_data_ext(
                    ctxPtr,
                    pin.AddrOfPinnedObject(),
                    (nuint)snapshot.Length,
                    seqId,
                    flags);

                if (restored == 0)
                    throw new RuntimeError($"Failed to restore recurrent/hybrid state snapshot for sequence {seqId}.");
            }
            finally
            {
                pin.Free();
            }
        }

        private static List<(int SeqId, int Token, int Pos)> ExtractBatchEntries(LLamaBatch batch)
        {
            var result = new List<(int SeqId, int Token, int Pos)>(batch.TokenCount);
            using var pin = batch.ToNativeBatch(out var nativeBatch);
            RawLlamaBatch raw = Unsafe.As<LLamaNativeBatch, RawLlamaBatch>(ref nativeBatch);

            if (raw.n_tokens <= 0 || raw.token == IntPtr.Zero || raw.pos == IntPtr.Zero)
                return result;

            for (int i = 0; i < raw.n_tokens; i++)
            {
                int token = Marshal.ReadInt32(raw.token, i * sizeof(int));
                int pos = Marshal.ReadInt32(raw.pos, i * sizeof(int));

                int seqId = 0;
                if (raw.seq_id != IntPtr.Zero && raw.n_seq_id != IntPtr.Zero)
                {
                    int nSeq = Marshal.ReadInt32(raw.n_seq_id, i * sizeof(int));
                    if (nSeq > 0)
                    {
                        IntPtr seqArrPtr = Marshal.ReadIntPtr(raw.seq_id, i * IntPtr.Size);
                        if (seqArrPtr != IntPtr.Zero)
                            seqId = Marshal.ReadInt32(seqArrPtr, 0);
                    }
                }

                result.Add((seqId, token, pos));
            }

            return result;
        }

        private void AssertNotDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(SpeculativeDecoder));
        }

        /// <summary>
        /// Disposes the speculative decoder, releasing owned samplers and native extended batches.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            _pendingH.Clear();
            _draftBatchExt?.Dispose();
            _draftSampler.Dispose();
            _targetSampler.Dispose();
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct RawLlamaBatch
        {
            public int n_tokens;
            public IntPtr token;
            public IntPtr embd;
            public IntPtr pos;
            public IntPtr n_seq_id;
            public IntPtr seq_id;
            public IntPtr logits;
        }

        private static class NativeMethods
        {
            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr llama_get_model(IntPtr ctx);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern uint llama_n_batch(IntPtr ctx);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern uint llama_n_ctx_seq(IntPtr ctx);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern int llama_model_n_embd_out(IntPtr model);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            public static extern bool llama_model_is_recurrent(IntPtr model);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            public static extern bool llama_model_is_hybrid(IntPtr model);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr llama_get_memory(IntPtr ctx);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            public static extern bool llama_memory_seq_rm(IntPtr mem, int seq_id, int p0, int p1);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern int llama_decode(IntPtr ctx, LLamaNativeBatch batch);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nuint llama_state_seq_get_size_ext(IntPtr ctx, int seq_id, uint flags);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nuint llama_state_seq_get_data_ext(IntPtr ctx, IntPtr dst, nuint size, int seq_id, uint flags);

            [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nuint llama_state_seq_set_data_ext(IntPtr ctx, IntPtr src, nuint size, int dest_seq_id, uint flags);
        }
    }
}
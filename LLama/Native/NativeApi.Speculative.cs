using System;
using System.Runtime.InteropServices;

namespace LLama.Native
{
    /// <summary>
    /// Processing mode passed to <see cref="NativeApi.llama_process"/>.
    /// </summary>
    public enum LLamaProcessType : int
    {
        /// <summary>
        /// Process the extended batch using the encoder graph.
        /// </summary>
        Encode = 0,

        /// <summary>
        /// Process the extended batch using the decoder graph.
        /// </summary>
        Decode = 1,
    }

    /// <summary>
    /// Native view of a contiguous embedding matrix passed into the extended batch API (<c>struct llama_embd</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLamaEmbd
    {
        /// <summary>
        /// Pointer to row-major float embedding data (<c>const float *</c>).
        /// </summary>
        public IntPtr data;

        /// <summary>
        /// Number of embedding rows in <see cref="data"/>.
        /// </summary>
        public nuint n_rows;

        /// <summary>
        /// Number of elements per embedding row.
        /// </summary>
        public nuint n_embd;
    }

    public static partial class NativeApi
    {
        #region NextN / Speculative Model & Context Queries

        /// <summary>
        /// Gets the configured number of recurrent-state snapshots per sequence for native rollback (0 = no rollback).
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <returns>The number of rollback snapshots per sequence.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_n_rs_seq(IntPtr ctx);

        /// <summary>
        /// Gets the number of appended NextN / MTP layers present in the loaded model.
        /// </summary>
        /// <param name="model">Pointer to a native <c>llama_model</c>.</param>
        /// <returns>The number of NextN layers (0 if the model has no MTP heads or <c>load_mtp</c> was disabled).</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_layer_nextn(IntPtr model);

        /// <summary>
        /// Gets the DFlash2 selector top-k candidate count per block position (0 if the model has no selector head).
        /// </summary>
        /// <param name="model">Pointer to a native <c>llama_model</c>.</param>
        /// <returns>The selector top-k value, or 0 if unsupported.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_dflash_selector_top_k(IntPtr model);

        /// <summary>
        /// Gets a native pointer (<c>const int32_t *</c>) to the target-model layer indices required by a draft model.
        /// </summary>
        /// <param name="model">Pointer to a native <c>llama_model</c>.</param>
        /// <returns>Pointer to the array of layer indices, or <see cref="IntPtr.Zero"/> if none.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_target_layer_ids(IntPtr model);

        /// <summary>
        /// Gets the number of extracted target-model layer IDs returned by <see cref="llama_model_target_layer_ids"/>.
        /// </summary>
        /// <param name="model">Pointer to a native <c>llama_model</c>.</param>
        /// <returns>The number of target layer IDs.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_model_target_layer_ids_n(IntPtr model);

        /// <summary>
        /// Returns the paired context configured via <c>llama_context_params::ctx_other</c> (e.g., the target context of an MTP draft context).
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <returns>Pointer to the paired <c>llama_context</c>, or <see cref="IntPtr.Zero"/> if none.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_ctx_other(IntPtr ctx);

        #endregion

        #region NextN / Speculative Decoding Context API

        /// <summary>
        /// Enables or disables NextN hidden-state embedding extraction for the context.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="value"><c>true</c> to capture NextN hidden states during decode.</param>
        /// <param name="masked">If <c>true</c>, only tokens with <c>batch.logits != 0</c> are captured; if <c>false</c>, all batch tokens are captured.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_embeddings_nextn(
            IntPtr ctx,
            [MarshalAs(UnmanagedType.I1)] bool value,
            [MarshalAs(UnmanagedType.I1)] bool masked);

        /// <summary>
        /// Selects which appended NextN head the <c>DECODER_MTP</c> graph executes (<c>il = n_layer + offset</c>). Default is 0.
        /// </summary>
        /// <param name="ctx">Pointer to a native MTP <c>llama_context</c>.</param>
        /// <param name="offset">Zero-based offset of the NextN layer to run.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_nextn_layer_offset(IntPtr ctx, int offset);

        /// <summary>
        /// Gets a pointer (<c>float *</c>) to all host-resident NextN embeddings produced by the last decode call.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <returns>Pointer to the contiguous NextN embeddings buffer, or <see cref="IntPtr.Zero"/> if unavailable.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_embeddings_nextn(IntPtr ctx);

        /// <summary>
        /// Gets a pointer (<c>float *</c>) to the NextN embedding row for the <paramref name="i"/>-th token of the last decode call.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="i">Token index (negative values index from the end, e.g., <c>-1</c> is the last token).</param>
        /// <returns>Pointer to the embedding row, or <see cref="IntPtr.Zero"/> if invalid.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_embeddings_nextn_ith(IntPtr ctx, int i);

        /// <summary>
        /// Enables or disables extraction of the input embeddings of layer <paramref name="lid"/>.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="lid">Zero-based layer index.</param>
        /// <param name="value"><c>true</c> to enable extraction; <c>false</c> to disable.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_embeddings_layer_inp(
            IntPtr ctx,
            uint lid,
            [MarshalAs(UnmanagedType.I1)] bool value);

        /// <summary>
        /// Gets a pointer (<c>float *</c>) to the extracted input embeddings of layer <paramref name="lid"/>.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="lid">Zero-based layer index.</param>
        /// <returns>Pointer to the layer input embeddings, or <see cref="IntPtr.Zero"/> if unavailable.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_embeddings_layer_inp(IntPtr ctx, uint lid);

        /// <summary>
        /// Configures whether enabled layer inputs are written to a device buffer (<c>[n_layer_inp * n_embd, n_batch]</c>) instead of host memory.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="value"><c>true</c> to use a device buffer.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_embeddings_layer_inp_dev(
            IntPtr ctx,
            [MarshalAs(UnmanagedType.I1)] bool value);

        /// <summary>
        /// Queries whether the context accepts extended batch entries added with <see cref="llama_batch_ext_add_embd_row"/>.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <returns><c>true</c> if device embedding rows are supported.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_supports_embd_row(IntPtr ctx);

        /// <summary>
        /// Configures whether unmasked <c>h_nextn</c> rows are written to a device buffer (<c>[n_embd_out, n_batch + n_seq_max]</c>) instead of host memory.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="value"><c>true</c> to enable device-resident NextN rows.</param>
        /// <returns><c>true</c> if the device buffer is active; <c>false</c> if rows will fall back to host memory.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_set_embeddings_nextn_dev(
            IntPtr ctx,
            [MarshalAs(UnmanagedType.I1)] bool value);

        /// <summary>
        /// Copies rows from the <c>h_nextn</c> device buffer into host memory.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="row0">Starting row index in the device buffer.</param>
        /// <param name="n_rows">Number of rows to copy.</param>
        /// <param name="out">Destination host buffer (<c>float *</c>).</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_get_embeddings_nextn_rows(
            IntPtr ctx,
            long row0,
            long n_rows,
            IntPtr @out);

        /// <summary>
        /// Copies rows from host memory into the <c>h_nextn</c> device buffer (e.g., scratch rows at <c>n_batch + seq_id</c>).
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="row0">Starting row index in the device buffer.</param>
        /// <param name="n_rows">Number of rows to copy.</param>
        /// <param name="data">Source host buffer (<c>const float *</c>).</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_set_embeddings_nextn_rows(
            IntPtr ctx,
            long row0,
            long n_rows,
            IntPtr data);

        /// <summary>
        /// Binds <paramref name="ctx"/> to read state rows (<see cref="llama_batch_ext_set_embd_state_row"/>) directly from the <c>h_nextn</c> device buffer of <paramref name="ctx_src"/>.
        /// </summary>
        /// <param name="ctx">Pointer to the receiving native <c>llama_context</c> (e.g., MTP draft context).</param>
        /// <param name="ctx_src">Pointer to the source native <c>llama_context</c>, or <see cref="IntPtr.Zero"/> to unbind.</param>
        /// <returns><c>true</c> if <paramref name="ctx"/> can decode batches using device state rows.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_set_ctx_state_src(IntPtr ctx, IntPtr ctx_src);

        #endregion

        #region Extended Batch API (llama_batch_ext)

        /// <summary>
        /// Allocates an opaque <c>llama_batch_ext</c> sized for the given context.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <returns>Pointer to the allocated <c>llama_batch_ext</c>.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_batch_ext_init(IntPtr ctx);

        /// <summary>
        /// Frees a <c>llama_batch_ext</c> previously allocated by <see cref="llama_batch_ext_init"/>.
        /// </summary>
        /// <param name="batch">Pointer to the <c>llama_batch_ext</c> to free.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_batch_ext_free(IntPtr batch);

        /// <summary>
        /// Clears all entries from the extended batch without deallocating its internal storage.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_batch_ext_clear(IntPtr batch);

        /// <summary>
        /// Adds an empty placeholder entry for <paramref name="seq_id"/> to the extended batch.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="seq_id">Sequence identifier.</param>
        /// <returns>The non-negative batch index on success, or a negative error code (<c>-1</c> full, <c>-2</c> invalid token, <c>-3</c> invalid seq_id).</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_batch_ext_add(IntPtr batch, int seq_id);

        /// <summary>
        /// Adds a token entry to the extended batch.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="seq_id">Sequence identifier.</param>
        /// <param name="id">Token ID.</param>
        /// <returns>The non-negative batch index on success, or a negative error code.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_batch_ext_add_token(IntPtr batch, int seq_id, int id);

        /// <summary>
        /// Adds an embedding entry to the extended batch.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="seq_id">Sequence identifier.</param>
        /// <param name="embd">Embedding descriptor.</param>
        /// <returns>The non-negative batch index on success, or a negative error code.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_batch_ext_add_embd(IntPtr batch, int seq_id, LLamaEmbd embd);

        /// <summary>
        /// Associates an additional sequence ID with the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index returned by an add call.</param>
        /// <param name="seq_id">Additional sequence identifier.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_add_seq(IntPtr batch, int idx, int seq_id);

        /// <summary>
        /// Attaches a token embedding to an existing entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="embd">Token embedding descriptor.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_embd_token(IntPtr batch, int idx, LLamaEmbd embd);

        /// <summary>
        /// Attaches a host-memory hidden-state embedding (e.g., MTP state) to the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="embd">State embedding descriptor.</param>
        /// <returns><c>true</c> on success; <c>false</c> if the context does not accept state embeddings.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_embd_state(IntPtr batch, int idx, LLamaEmbd embd);

        /// <summary>
        /// Adds an entry referencing row <paramref name="row"/> of the paired context's layer-input device buffer (DFlash/DSpark).
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="seq_id">Sequence identifier.</param>
        /// <param name="row">Row index in <c>ctx_other</c>'s layer-input device buffer.</param>
        /// <returns>The non-negative batch index on success, or a negative error code.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_batch_ext_add_embd_row(IntPtr batch, int seq_id, int row);

        /// <summary>
        /// Binds the entry at <paramref name="idx"/> to row <paramref name="row"/> of the <c>h_nextn</c> device buffer bound via <see cref="llama_set_ctx_state_src"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="row">Row index in the source context's <c>h_nextn</c> device buffer.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_embd_state_row(IntPtr batch, int idx, int row);

        /// <summary>
        /// Sets whether output embeddings should be generated for the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="value"><c>true</c> to request output embeddings.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_output_embd(
            IntPtr batch,
            int idx,
            [MarshalAs(UnmanagedType.I1)] bool value);

        /// <summary>
        /// Sets whether output logits should be generated for the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="value"><c>true</c> to request output logits.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_output_logits(
            IntPtr batch,
            int idx,
            [MarshalAs(UnmanagedType.I1)] bool value);

        /// <summary>
        /// Sets a single scalar position for the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="pos">Token position.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_pos(IntPtr batch, int idx, ref int pos);

        /// <summary>
        /// Sets multi-dimensional positions (e.g., M-RoPE <c>[t, y, x, z]</c>) for the entry at <paramref name="idx"/>.
        /// </summary>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <param name="idx">Entry index.</param>
        /// <param name="pos">Pointer to <c>const llama_pos *</c>.</param>
        /// <returns><c>true</c> on success.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_batch_ext_set_pos(IntPtr batch, int idx, IntPtr pos);

        /// <summary>
        /// Executes an extended batch against <paramref name="ctx"/>. Return codes match <c>llama_decode</c>.
        /// </summary>
        /// <param name="ctx">Pointer to a native <c>llama_context</c>.</param>
        /// <param name="type">Encode or decode mode.</param>
        /// <param name="batch">Pointer to a native <c>llama_batch_ext</c>.</param>
        /// <returns><c>0</c> on success; positive warning code or negative error code otherwise.</returns>
        [DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_process(IntPtr ctx, LLamaProcessType type, IntPtr batch);

        #endregion
    }
}
using LLama.Common;
using LLama.Native;
using Spectre.Console;
using System.Diagnostics;
using System.Text;

namespace LLama.Examples.Examples
{
    /// <summary>
    /// A utility class to benchmark and compare the Tokens-Per-Second (TPS) performance of 
    /// Standard Autoregressive generation versus Speculative Decoding (Dual-Model and MTP).
    /// </summary>
    public static class SpeculativeBenchmark
    {
        /// <summary>
        /// Captures timing, throughput, draft acceptance statistics, and generated text for a single benchmark pass.
        /// </summary>
        /// <param name="Name">Display label for the decoding strategy.</param>
        /// <param name="TotalTokens">Total number of tokens generated.</param>
        /// <param name="ElapsedSeconds">Total elapsed wall-clock time in seconds.</param>
        /// <param name="TokensPerSecond">Average throughput in tokens per second.</param>
        /// <param name="AcceptanceRate">Fraction of proposed draft tokens accepted (0.0 to 1.0).</param>
        /// <param name="DraftTokensAccepted">Total proposed draft tokens accepted by the target model.</param>
        /// <param name="DraftTokensProposed">Total draft tokens proposed during the run.</param>
        /// <param name="Output">Full generated text produced during the benchmark pass.</param>
        public record BenchmarkResult(
            string Name,
            int TotalTokens,
            double ElapsedSeconds,
            double TokensPerSecond,
            double AcceptanceRate,
            int DraftTokensAccepted = 0,
            int DraftTokensProposed = 0,
            string Output = ""
        );

        public static async Task RunAsync(
            string targetModelPath,
            string draftModelPath,
            string prompt = "The following is a list of numbers from 1 to 500: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,",
            int maxTokens = 128,
            int draftTokens = 16,
            bool useMtp = false)
        {
            // Check if the user provided a distinct draft file (prevents accidental double-loading)
            bool isSeparateDraft = !string.Equals(targetModelPath, draftModelPath, StringComparison.OrdinalIgnoreCase);
            string draftDisplay = useMtp
                ? (isSeparateDraft ? $"[Split MTP] {draftModelPath}" : "[Bundled MTP Self-Drafting]")
                : draftModelPath;

            Console.WriteLine("==========================================================");
            Console.WriteLine("         LLAMASHARP SPECULATIVE DECODING BENCHMARK        ");
            Console.WriteLine("==========================================================");
            Console.WriteLine($"Target Model : {targetModelPath}");
            Console.WriteLine($"Draft Model  : {draftDisplay}");
            Console.WriteLine($"Draft Budget : {draftTokens} tokens per burst");
            Console.WriteLine($"Target Tokens: {maxTokens} tokens");
            Console.WriteLine("----------------------------------------------------------\n");

            LLamaWeights? draftWeights = null;

            if (draftTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(draftTokens), "Draft token budget must be greater than zero.");

            // First load the draft model that it can be fully loaded into the GPU.
            // Then, load the target model and distribute the layers as needed (GPUs, COU).

            #region draft model
            // Always use draftModelPath here so Gemma 4 loads the correct assistant GGUF
            var draftParams = new ModelParams(draftModelPath)
            {
                ContextSize = 2048, //Most of the time, the next word in a sentence depends almost entirely on the immediately preceding words (grammar, formatting, or the current train of thought). A draft model only needs a few hundred tokens of context to successfully draft the next words...
                BatchSize = 512,
                LoadMTP = useMtp,
                GpuLayerCount = -1,
                MainGpu = 0,
                UBatchSize = 512,
                SplitMode = GPUSplitMode.None,
                SeqMax = (uint)(draftTokens + 1),
                // In MTP mode, the secondary context MUST be explicitly flagged as LLamaContextType.Mtp
                ContextType = useMtp ? LLamaContextType.Mtp : LLamaContextType.Default,

                // NATIVE PERFORMANCE OPTIMIZERS
                KVUnified = false,
                SwaFull = true
            };
            // Important: In MTP mode, the executor will internally re-use the target weights for the draft context

            // Only load separate draft weights if the paths are actually different.
            // This supports Dual-Model AND split-MTP (Gemma 4), while keeping bundled-MTP (Qwen) safe.
            if (isSeparateDraft)
            {
                draftWeights = LLamaWeights.LoadFromFile(draftParams);
            }
            #endregion

            #region target model
            var targetParams = new ModelParams(targetModelPath)
            {
                ContextSize = 4096,
                BatchSize = 512,
                LoadMTP = useMtp,
                GpuLayerCount = 20,
                MainGpu = 0,
                UBatchSize = 512,
                SplitMode = GPUSplitMode.Layer,
                SeqMax = (uint)(draftTokens + 1),
                ContextType = LLamaContextType.Default,

                // NATIVE PERFORMANCE OPTIMIZERS
                // llama.cpp explicitly warns that KVUnified=true can cause bad performance when SeqMax > 1
                KVUnified = false,
                SwaFull = true
            };

            using var targetWeights = LLamaWeights.LoadFromFile(targetParams);

            int detectedMtpHeads = 0;

            // MTP AUTO-DETECT (works for both bundled MTP in targetWeights and split MTP in draftWeights)
            if (useMtp)
            {
                detectedMtpHeads = NativeApi.llama_model_n_layer_nextn((draftWeights ?? targetWeights).NativeHandle.DangerousGetHandle());
                if (detectedMtpHeads <= 0 && draftWeights != null)
                    detectedMtpHeads = NativeApi.llama_model_n_layer_nextn(targetWeights.NativeHandle.DangerousGetHandle());

                if (detectedMtpHeads <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("\n[Error] MTP mode was requested, but neither the target nor draft GGUF exposes NextN/MTP layers (llama_model_n_layer_nextn == 0).");
                    Console.WriteLine("        Please select an MTP-enabled GGUF (e.g., Qwen3.5-4B-MTP-Q4_K_M.gguf or Gemma-4 + mtp-gemma-4),");
                    Console.WriteLine("        or switch to Dual-Model speculation with a smaller same-family draft model.");
                    Console.ResetColor();

                    draftWeights?.Dispose();
                    return;
                }

                Console.WriteLine($"[Info] Detected {detectedMtpHeads} NextN/MTP head(s).");
            }
            else if (draftWeights != null)
            {
                // Early sanity check for Dual-Model tokenizer & vocabulary compatibility
                targetWeights.Metadata.TryGetValue("tokenizer.ggml.model", out var tgtTok);
                draftWeights.Metadata.TryGetValue("tokenizer.ggml.model", out var dftTok);
                targetWeights.Metadata.TryGetValue("tokenizer.ggml.pre", out var tgtPre);
                draftWeights.Metadata.TryGetValue("tokenizer.ggml.pre", out var dftPre);

                int tgtVocabSize = targetWeights.Vocab.Count;
                int dftVocabSize = draftWeights.Vocab.Count;

                bool mismatch =
                    !string.Equals(tgtTok, dftTok, StringComparison.OrdinalIgnoreCase) ||
                    !string.Equals(tgtPre, dftPre, StringComparison.OrdinalIgnoreCase) ||
                    tgtVocabSize != dftVocabSize;

                if (mismatch)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"\n[Warning] Tokenizer/vocab mismatch detected!");
                    Console.WriteLine($"          Target: model='{tgtTok}', pre='{tgtPre}', vocab={tgtVocabSize}");
                    Console.WriteLine($"          Draft : model='{dftTok}', pre='{dftPre}', vocab={dftVocabSize}");
                    Console.WriteLine("          Dual-Model speculation requires target and draft models from the same tokenizer family.");
                    Console.ResetColor();
                }
            }
            #endregion

            try
            {
                Console.WriteLine("[1/3] Warming up model & JIT compiler...");
                var warmupExecutor = new StatelessExecutor(targetWeights, targetParams);
                var warmupParams = new InferenceParams { MaxTokens = 10 };
                await foreach (var _ in warmupExecutor.InferAsync("Warmup test.", warmupParams)) { }
                Console.WriteLine("      Warmup complete.\n");

                Console.WriteLine($"[2/3] Running Speculative Decoding ({(useMtp ? "MTP" : "Draft-Simple")})...");
                // Important: In MTP mode, the executor will internally re-use the target weights for the 
                // draft context (draftWeights = null), but draftParams MUST be passed to trigger MTP context creation.
                var specExecutor = new StatelessExecutor(
                    weights: targetWeights,
                    @params: targetParams,
                    draftWeights: draftWeights,
                    draftParams: draftParams,
                    draftTokens: draftTokens,
                    useMtp: useMtp
                );

                var speculativeResult = await MeasureInferenceAsync(useMtp ? "MTP Speculative" : "Draft-Simple Speculative", specExecutor, prompt, maxTokens);

                Console.WriteLine("[3/3] Running Baseline: Standard Autoregressive Generation...");
                var standardExecutor = new StatelessExecutor(
                    weights: targetWeights,
                    @params: targetParams,
                    draftWeights: null,
                    draftParams: null,
                    draftTokens: 0,
                    useMtp: false
                );

                var baselineResult = await MeasureInferenceAsync("Standard Autoregressive", standardExecutor, prompt, maxTokens);

                string modeLabel = useMtp
                    ? (isSeparateDraft ? "Split MTP" : "Bundled MTP")
                    : "Dual-Model (Draft-Simple)";

                PrintFullReport(
                    modeLabel: modeLabel,
                    targetModelPath: targetModelPath,
                    draftDisplay: draftDisplay,
                    detectedMtpHeads: detectedMtpHeads,
                    targetParams: targetParams,
                    draftParams: draftParams,
                    draftTokens: draftTokens,
                    maxTokens: maxTokens,
                    prompt: prompt,
                    baseline: baselineResult,
                    speculative: speculativeResult);
            }
            finally
            {
                draftWeights?.Dispose();
            }
        }

        /// <summary>
        /// Runs an interactive console wizard to configure and execute a speculative decoding benchmark.
        /// </summary>
        public static async Task RunInteractiveAsync()
        {
            const string ModeDualModel = "1. Dual-Model (Large Target + Smaller Same-Family Draft — e.g. Llama-3-8B + Llama-3.2-1B, Qwen3.6-35B + Qwen3.5-0.8B)";
            const string ModeBundledMtp = "2. Bundled MTP (Single GGUF with built-in NextN heads — e.g. Qwen3.5-4B-MTP, DeepSeek-R1-MTP)";
            const string ModeSplitMtp = "3. Split MTP (Target GGUF + Separate MTP Head GGUF — e.g. gemma-4-12b-it + mtp-gemma-4-12b-it)";

            Spectre.Console.AnsiConsole.MarkupLine("[bold cyan]=== Speculative Decoding Benchmark ===[/]");
            Spectre.Console.AnsiConsole.MarkupLine("[grey]Compares Standard Autoregressive decoding against Speculative Decoding using your selected model setup.[/]\n");

            // 1. Choose speculation mode FIRST so the user knows which GGUF files are expected
            var mode = Spectre.Console.AnsiConsole.Prompt(
                new Spectre.Console.SelectionPrompt<string>()
                    .Title("[bold]Select Speculative Decoding Mode:[/]")
                    .AddChoices(ModeDualModel, ModeBundledMtp, ModeSplitMtp));

            bool useMtp = mode != ModeDualModel;
            bool needsSeparateDraft = mode != ModeBundledMtp;

            // 2. Prompt for Target Model with mode-specific guidance
            Spectre.Console.AnsiConsole.MarkupLine(mode switch
            {
                ModeDualModel => "\n[bold]Step 1/2 — Target Model (Large Primary Model):[/]\n[grey]Example: Meta-Llama-3-8B-Instruct-Q8_0.gguf or Qwen3.6-35B-A3B-UD-Q5_K_M.gguf[/]",
                ModeBundledMtp => "\n[bold]Step 1/1 — Target MTP Model (Single GGUF with NextN heads):[/]\n[grey]Example: Qwen3.5-4B-MTP-Q4_K_M.gguf[/]",
                _ => "\n[bold]Step 1/2 — Target Model (Primary Model):[/]\n[grey]Example: gemma-4-12b-it-UD-Q4_K_XL.gguf[/]"
            });
            var targetModelPath = UserSettings.GetSpeculativeTargetModelPath();

            // 3. Prompt for Draft / Split-MTP Model only when required by the selected mode
            string draftModelPath = targetModelPath;
            if (needsSeparateDraft)
            {
                if (mode == ModeDualModel)
                {
                    Spectre.Console.AnsiConsole.MarkupLine("\n[bold]Step 2/2 — Draft Model (Smaller Model with SAME Tokenizer/Vocab as Target):[/]");
                    Spectre.Console.AnsiConsole.MarkupLine("[grey]Example: Llama-3.2-1B-Instruct-Q4_0.gguf (for Llama-3) or Qwen3.5-0.8B / Qwen3.5-2B (for Qwen3.5/3.6)[/]");
                    draftModelPath = UserSettings.GetSpeculativeDraftModelPath("DRAFT model.gguf (smaller same-vocab model)");
                }
                else
                {
                    Spectre.Console.AnsiConsole.MarkupLine("\n[bold]Step 2/2 — Separate MTP Head Model:[/]");
                    Spectre.Console.AnsiConsole.MarkupLine("[grey]Example: mtp-gemma-4-12b-it-F16.gguf[/]");
                    draftModelPath = UserSettings.GetSpeculativeDraftModelPath("MTP head model.gguf");
                }
            }

            // 4. Ask for burst budget & token count with practical defaults
            int defaultDraftTokens = mode switch
            {
                ModeBundledMtp => 1,
                ModeSplitMtp => 4,
                _ => 5
            };

            var draftTokens = Spectre.Console.AnsiConsole.Ask<int>(
                "Enter draft tokens budget per burst (recommended: 4-6 Dual-Model, 1-4 MTP):",
                defaultDraftTokens);

            var maxTokens = Spectre.Console.AnsiConsole.Ask<int>(
                "Enter max tokens to generate for the benchmark:",
                128);

            // 5. Run the benchmark
            await RunAsync(
                targetModelPath: targetModelPath,
                draftModelPath: draftModelPath,
                maxTokens: maxTokens,
                draftTokens: draftTokens,
                useMtp: useMtp
            );
        }

        private static async Task<BenchmarkResult> MeasureInferenceAsync(string testName, StatelessExecutor executor, string prompt, int maxTokens)
        {
            var inferenceParams = new InferenceParams { MaxTokens = maxTokens };
            var outputBuilder = new StringBuilder();
            int tokenCount = 0;
            var stopwatch = Stopwatch.StartNew();

            await foreach (var token in executor.InferAsync(prompt, inferenceParams))
            {
                outputBuilder.Append(token);
                tokenCount++;
            }

            stopwatch.Stop();
            double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
            double tps = elapsedSeconds > 0 ? tokenCount / elapsedSeconds : 0;

            return new BenchmarkResult(
                Name: testName,
                TotalTokens: tokenCount,
                ElapsedSeconds: elapsedSeconds,
                TokensPerSecond: tps,
                AcceptanceRate: executor.AcceptanceRate,
                DraftTokensAccepted: executor.TotalDraftTokensAccepted,
                DraftTokensProposed: executor.TotalDraftTokensProposed,
                Output: outputBuilder.ToString());
        }

        /// <summary>
        /// Prints a consolidated benchmark report including models, configuration options, performance metrics, input prompt, and generated outputs.
        /// </summary>
        private static void PrintFullReport(
            string modeLabel,
            string targetModelPath,
            string draftDisplay,
            int detectedMtpHeads,
            ModelParams targetParams,
            ModelParams draftParams,
            int draftTokens,
            int maxTokens,
            string prompt,
            BenchmarkResult baseline,
            BenchmarkResult speculative)
        {
            double speedupRatio = baseline.TokensPerSecond > 0
                ? speculative.TokensPerSecond / baseline.TokensPerSecond
                : 0;
            double percentageGain = (speedupRatio - 1.0) * 100.0;
            bool exactOutputMatch = string.Equals(baseline.Output, speculative.Output, StringComparison.Ordinal);

            string divider = new string('=', 80);
            string subDivider = new string('-', 80);

            Console.WriteLine();
            Console.WriteLine(divider);
            Console.WriteLine("                  FULL SPECULATIVE BENCHMARK REPORT                             ");
            Console.WriteLine(divider);

            Console.WriteLine("MODELS & MODE");
            Console.WriteLine($"  Mode         : {modeLabel}");
            Console.WriteLine($"  Target Model : {targetModelPath}");
            Console.WriteLine($"  Draft Model  : {draftDisplay}");
            if (detectedMtpHeads > 0)
                Console.WriteLine($"  MTP Heads    : {detectedMtpHeads}");

            Console.WriteLine(subDivider);
            Console.WriteLine("OPTIONS USED");
            Console.WriteLine($"  Draft Budget : {draftTokens} tokens/burst | Max Tokens: {maxTokens}");
            Console.WriteLine($"  Target Params: Ctx={targetParams.ContextSize}, Batch={targetParams.BatchSize}/{targetParams.UBatchSize}, " +
                              $"GpuLayers={targetParams.GpuLayerCount}, MainGpu={targetParams.MainGpu}, Split={targetParams.SplitMode}, " +
                              $"SeqMax={targetParams.SeqMax}, KVUnified={targetParams.KVUnified}, SwaFull={targetParams.SwaFull}, LoadMTP={targetParams.LoadMTP}");
            Console.WriteLine($"  Draft Params : Ctx={draftParams.ContextSize}, Batch={draftParams.BatchSize}/{draftParams.UBatchSize}, " +
                              $"GpuLayers={draftParams.GpuLayerCount}, MainGpu={draftParams.MainGpu}, Split={draftParams.SplitMode}, " +
                              $"SeqMax={draftParams.SeqMax}, CtxType={draftParams.ContextType}, KVUnified={draftParams.KVUnified}, SwaFull={draftParams.SwaFull}, LoadMTP={draftParams.LoadMTP}");

            Console.WriteLine(subDivider);
            Console.WriteLine("PERFORMANCE RESULTS");
            Console.WriteLine($"{"Strategy",-26} | {"Tokens",-6} | {"Time (s)",-8} | {"Tok/s",-8} | {"Drafts",-11} | {"Accept %",-8}");
            Console.WriteLine(subDivider);
            Console.WriteLine($"{baseline.Name,-26} | {baseline.TotalTokens,-6} | {baseline.ElapsedSeconds,-8:F2} | {baseline.TokensPerSecond,-8:F2} | {"N/A",-11} | {"N/A",-8}");
            Console.WriteLine($"{speculative.Name,-26} | {speculative.TotalTokens,-6} | {speculative.ElapsedSeconds,-8:F2} | {speculative.TokensPerSecond,-8:F2} | " +
                              $"{$"{speculative.DraftTokensAccepted}/{speculative.DraftTokensProposed}",-11} | {speculative.AcceptanceRate * 100,6:F1}%");
            Console.WriteLine(subDivider);

            if (speedupRatio >= 1.0)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  Performance  : {speedupRatio:F2}x Speedup (+{percentageGain:F1}%)");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  Performance  : {speedupRatio:F2}x Slowdown ({percentageGain:F1}%)");
            }
            Console.ResetColor();
            Console.WriteLine($"  Output Parity: {(exactOutputMatch ? "EXACT MATCH (Baseline == Speculative)" : "DIFFERS (check sampling/token limit)")}");

            Console.WriteLine(subDivider);
            Console.WriteLine("INPUT PROMPT");
            Console.WriteLine(prompt);

            Console.WriteLine(subDivider);
            Console.WriteLine("OUTPUT — STANDARD AUTOREGRESSIVE");
            Console.WriteLine(baseline.Output);

            Console.WriteLine(subDivider);
            Console.WriteLine($"OUTPUT — {speculative.Name.ToUpperInvariant()}");
            Console.WriteLine(speculative.Output);
            Console.WriteLine(divider);
            Console.WriteLine();
        }
    }
}
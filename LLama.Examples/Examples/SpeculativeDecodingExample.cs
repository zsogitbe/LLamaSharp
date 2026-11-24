using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using LLama.Common;
using LLama.Native;
using Spectre.Console;

namespace LLama.Examples.Examples
{
    /// <summary>
    /// Demonstrates interactive inference accelerated by Speculative Decoding (Dual-Model Draft-Simple)
    /// and Multi-Token Prediction (Bundled or Split MTP) using <see cref="StatelessExecutor"/>.
    /// </summary>
    public class SpeculativeDecodingExample
    {
        /// <summary>
        /// Captures the input, generated output, and performance metrics for a single interactive chat turn.
        /// </summary>
        private sealed record TurnReport(
            int TurnNumber,
            string Prompt,
            string Output,
            int TokenCount,
            double ElapsedSeconds,
            double TokensPerSecond,
            int DraftTokensAccepted,
            int DraftTokensProposed,
            double AcceptanceRate);

        /// <summary>
        /// Runs the interactive speculative decoding chat example.
        /// </summary>
        public static async Task Run()
        {
            const string ModeDualModel = "1. Dual-Model (Large Target + Smaller Same-Family Draft — e.g. Llama-3-8B + Llama-3.2-1B)";
            const string ModeBundledMtp = "2. Bundled MTP (Single GGUF with built-in NextN heads — e.g. Qwen3.5-4B-MTP, DeepSeek-R1-MTP)";
            const string ModeSplitMtp = "3. Split MTP (Target GGUF + Separate MTP Head GGUF — e.g. gemma-4-12b-it + mtp-gemma-4-12b-it)";

            AnsiConsole.MarkupLine("[bold cyan]=== Speculative Decoding & Multi-Token Prediction (MTP) ===[/]");
            AnsiConsole.MarkupLine("[grey]Proposes multiple draft tokens per step and verifies them in a single target pass for lossless greedy acceleration.[/]\n");

            // 1. Choose speculation mode first so model requirements are clear
            var mode = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[bold]Select Speculative Decoding Mode:[/]")
                    .AddChoices(ModeDualModel, ModeBundledMtp, ModeSplitMtp));

            bool useMtp = mode != ModeDualModel;
            bool needsSeparateDraft = mode != ModeBundledMtp;

            // 2. Prompt for Target Model
            AnsiConsole.MarkupLine(mode switch
            {
                ModeDualModel => "\n[bold]Step 1/2 — Target Model (Large Primary Model):[/]\n[grey]Example: Meta-Llama-3-8B-Instruct-Q8_0.gguf or Qwen3.6-35B-A3B-UD-Q5_K_M.gguf[/]",
                ModeBundledMtp => "\n[bold]Step 1/1 — Target MTP Model (Single GGUF with NextN heads):[/]\n[grey]Example: Qwen3.5-4B-MTP-Q4_K_M.gguf[/]",
                _ => "\n[bold]Step 1/2 — Target Model (Primary Model):[/]\n[grey]Example: gemma-4-12b-it-UD-Q4_K_XL.gguf[/]"
            });
            string targetModelPath = UserSettings.GetSpeculativeTargetModelPath();

            // 3. Prompt for Draft / Split-MTP Model when required
            string draftModelPath = targetModelPath;
            if (needsSeparateDraft)
            {
                if (mode == ModeDualModel)
                {
                    AnsiConsole.MarkupLine("\n[bold]Step 2/2 — Draft Model (Smaller Model with SAME Tokenizer/Vocab as Target):[/]");
                    AnsiConsole.MarkupLine("[grey]Example: Llama-3.2-1B-Instruct-Q4_0.gguf (for Llama-3) or Qwen3.5-0.8B / Qwen3.5-2B (for Qwen3.5/3.6)[/]");
                    draftModelPath = UserSettings.GetSpeculativeDraftModelPath("DRAFT model.gguf (smaller same-vocab model)");
                }
                else
                {
                    AnsiConsole.MarkupLine("\n[bold]Step 2/2 — Separate MTP Head Model:[/]");
                    AnsiConsole.MarkupLine("[grey]Example: mtp-gemma-4-12b-it-F16.gguf[/]");
                    draftModelPath = UserSettings.GetSpeculativeDraftModelPath("MTP head model.gguf");
                }
            }

            // 4. Draft burst budget
            int defaultDraftTokens = mode switch
            {
                ModeBundledMtp => 1,
                ModeSplitMtp => 4,
                _ => 5
            };
            int draftTokens = AnsiConsole.Ask<int>(
                "Enter draft tokens budget per burst (recommended: 4-6 Dual-Model, 1-4 MTP):",
                defaultDraftTokens);

            if (draftTokens <= 0)
            {
                AnsiConsole.MarkupLine("[red]Draft token budget must be greater than 0.[/]");
                return;
            }

            // 5. Configure Target and Draft ModelParams
            bool isSeparateDraft = !string.Equals(targetModelPath, draftModelPath, StringComparison.OrdinalIgnoreCase);

            var targetParams = new ModelParams(targetModelPath)
            {
                ContextSize = 2048,
                LoadMTP = useMtp,
                GpuLayerCount = -1,
                MainGpu = 0,
                ContextType = LLamaContextType.Default
            };

            // In MTP mode, StatelessExecutor pairs this context to the target context via CtxOther automatically.
            var draftParams = new ModelParams(draftModelPath)
            {
                ContextSize = 2048,
                LoadMTP = useMtp,
                GpuLayerCount = -1,
                MainGpu = 0,
                ContextType = useMtp ? LLamaContextType.Mtp : LLamaContextType.Default
            };

            LLamaWeights? draftWeights = null;
            try
            {
                if (isSeparateDraft)
                {
                    AnsiConsole.MarkupLine("\n[grey]Loading Draft Model...[/]");
                    draftWeights = LLamaWeights.LoadFromFile(draftParams);
                }

                AnsiConsole.MarkupLine("[grey]Loading Target Model...[/]");
                using var targetWeights = LLamaWeights.LoadFromFile(targetParams);

                int detectedMtpHeads = 0;

                // Validate MTP capability before entering the chat loop
                if (useMtp)
                {
                    detectedMtpHeads = Math.Max(
                        NativeApi.llama_model_n_layer_nextn((draftWeights ?? targetWeights).NativeHandle.DangerousGetHandle()),
                        NativeApi.llama_model_n_layer_nextn(targetWeights.NativeHandle.DangerousGetHandle()));

                    if (detectedMtpHeads <= 0)
                    {
                        AnsiConsole.MarkupLine("[red]ERROR: Selected model does not expose NextN/MTP layers (llama_model_n_layer_nextn == 0).[/]");
                        AnsiConsole.MarkupLine("[grey]Use an MTP-enabled GGUF or switch to Dual-Model speculation.[/]");
                        return;
                    }

                    AnsiConsole.MarkupLine($"[grey]Detected {detectedMtpHeads} NextN/MTP head(s).[/]");
                }

                var executor = new StatelessExecutor(
                    weights: targetWeights,
                    @params: targetParams,
                    draftWeights: draftWeights,
                    draftParams: draftParams,
                    draftTokens: draftTokens,
                    useMtp: useMtp
                )
                {
                    ApplyTemplate = true // Formats prompts automatically for Instruct models
                };

                string modeLabel = useMtp
                    ? (isSeparateDraft ? "Split MTP" : "Bundled MTP")
                    : "Dual-Model (Draft-Simple)";
                string draftDisplay = useMtp
                    ? (isSeparateDraft ? $"[Split MTP] {draftModelPath}" : "[Bundled MTP Self-Drafting]")
                    : draftModelPath;

                await RunChatLoop(
                    executor: executor,
                    modeLabel: modeLabel,
                    targetModelPath: targetModelPath,
                    draftDisplay: draftDisplay,
                    detectedMtpHeads: detectedMtpHeads,
                    targetParams: targetParams,
                    draftParams: draftParams,
                    draftTokens: draftTokens);
            }
            finally
            {
                draftWeights?.Dispose();
            }
        }

        private static async Task RunChatLoop(
            StatelessExecutor executor,
            string modeLabel,
            string targetModelPath,
            string draftDisplay,
            int detectedMtpHeads,
            ModelParams targetParams,
            ModelParams draftParams,
            int draftTokens)
        {
            var inferenceParams = new InferenceParams { MaxTokens = 256 };
            var turns = new List<TurnReport>();

            AnsiConsole.MarkupLine($"\n[bold green]Ready![/] Mode: [cyan]{modeLabel}[/] (draft budget: {draftTokens}). Type [bold]exit[/] to quit and view the full session report.");
            AnsiConsole.MarkupLine("[grey]Each reply is followed by throughput and draft acceptance statistics (stateless: each prompt runs independently).[/]");

            while (true)
            {
                string prompt = AnsiConsole.Ask<string>("\n[green]User:[/]");
                if (prompt.Equals("exit", StringComparison.OrdinalIgnoreCase))
                    break;

                AnsiConsole.Markup("[yellow]Bot: [/]");

                var outputBuilder = new StringBuilder();
                int tokenCount = 0;
                var sw = Stopwatch.StartNew();

                // Speculative decoding works transparently with the standard streaming API.
                // Use Console.Write (not AnsiConsole.Markup) so model output containing '[' or ']' never throws.
                await foreach (var token in executor.InferAsync(prompt, inferenceParams))
                {
                    Console.Write(token);
                    outputBuilder.Append(token);
                    tokenCount++;
                }

                sw.Stop();
                Console.WriteLine();

                double elapsed = sw.Elapsed.TotalSeconds;
                double tps = elapsed > 0 ? tokenCount / elapsed : 0;
                AnsiConsole.MarkupLine(
                    $"[dim][[{modeLabel} | {tokenCount} tokens in {elapsed:F2}s ({tps:F1} tok/s) | " +
                    $"Drafts accepted: {executor.TotalDraftTokensAccepted}/{executor.TotalDraftTokensProposed} ({executor.AcceptanceRate * 100:F1}%)]][/]");

                turns.Add(new TurnReport(
                    TurnNumber: turns.Count + 1,
                    Prompt: prompt,
                    Output: outputBuilder.ToString(),
                    TokenCount: tokenCount,
                    ElapsedSeconds: elapsed,
                    TokensPerSecond: tps,
                    DraftTokensAccepted: executor.TotalDraftTokensAccepted,
                    DraftTokensProposed: executor.TotalDraftTokensProposed,
                    AcceptanceRate: executor.AcceptanceRate));
            }

            PrintSessionReport(
                modeLabel: modeLabel,
                targetModelPath: targetModelPath,
                draftDisplay: draftDisplay,
                detectedMtpHeads: detectedMtpHeads,
                targetParams: targetParams,
                draftParams: draftParams,
                draftTokens: draftTokens,
                maxTokens: inferenceParams.MaxTokens,
                applyTemplate: executor.ApplyTemplate,
                turns: turns);
        }

        /// <summary>
        /// Prints a complete summary report at the end of the interactive session with models, options, totals, and all prompt/output transcripts.
        /// </summary>
        private static void PrintSessionReport(
            string modeLabel,
            string targetModelPath,
            string draftDisplay,
            int detectedMtpHeads,
            ModelParams targetParams,
            ModelParams draftParams,
            int draftTokens,
            int maxTokens,
            bool applyTemplate,
            IReadOnlyList<TurnReport> turns)
        {
            string divider = new string('=', 80);
            string subDivider = new string('-', 80);

            Console.WriteLine();
            Console.WriteLine(divider);
            Console.WriteLine("                FULL SPECULATIVE DECODING SESSION REPORT                        ");
            Console.WriteLine(divider);

            Console.WriteLine("MODELS & MODE");
            Console.WriteLine($"  Mode         : {modeLabel}");
            Console.WriteLine($"  Target Model : {targetModelPath}");
            Console.WriteLine($"  Draft Model  : {draftDisplay}");
            if (detectedMtpHeads > 0)
                Console.WriteLine($"  MTP Heads    : {detectedMtpHeads}");

            Console.WriteLine(subDivider);
            Console.WriteLine("OPTIONS USED");
            Console.WriteLine($"  Draft Budget : {draftTokens} tokens/burst | Max Tokens: {maxTokens} | ApplyTemplate: {applyTemplate}");
            Console.WriteLine($"  Target Params: Ctx={targetParams.ContextSize}, Batch={targetParams.BatchSize}/{targetParams.UBatchSize}, " +
                              $"GpuLayers={targetParams.GpuLayerCount}, MainGpu={targetParams.MainGpu}, Split={targetParams.SplitMode?.ToString() ?? "Default"}, " +
                              $"LoadMTP={targetParams.LoadMTP}");
            Console.WriteLine($"  Draft Params : Ctx={draftParams.ContextSize}, Batch={draftParams.BatchSize}/{draftParams.UBatchSize}, " +
                              $"GpuLayers={draftParams.GpuLayerCount}, MainGpu={draftParams.MainGpu}, Split={draftParams.SplitMode?.ToString() ?? "Default"}, " +
                              $"CtxType={draftParams.ContextType}, LoadMTP={draftParams.LoadMTP}");

            Console.WriteLine(subDivider);
            Console.WriteLine("SESSION SUMMARY");
            if (turns.Count == 0)
            {
                Console.WriteLine("  No prompts were executed.");
                Console.WriteLine(divider);
                Console.WriteLine();
                return;
            }

            int totalTokens = 0;
            double totalTime = 0;
            int totalAccepted = 0;
            int totalProposed = 0;
            foreach (var turn in turns)
            {
                totalTokens += turn.TokenCount;
                totalTime += turn.ElapsedSeconds;
                totalAccepted += turn.DraftTokensAccepted;
                totalProposed += turn.DraftTokensProposed;
            }

            double avgTps = totalTime > 0 ? totalTokens / totalTime : 0;
            double overallAccept = totalProposed > 0 ? (double)totalAccepted / totalProposed * 100.0 : 0;
            Console.WriteLine($"  Turns        : {turns.Count}");
            Console.WriteLine($"  Total Output : {totalTokens} tokens in {totalTime:F2}s ({avgTps:F2} tok/s avg)");
            Console.WriteLine($"  Drafts       : {totalAccepted}/{totalProposed} accepted ({overallAccept:F1}%)");

            Console.WriteLine(subDivider);
            Console.WriteLine("TRANSCRIPT (INPUT & OUTPUT)");
            foreach (var turn in turns)
            {
                Console.WriteLine($"[Turn {turn.TurnNumber}] {turn.TokenCount} tokens | {turn.ElapsedSeconds:F2}s ({turn.TokensPerSecond:F1} tok/s) | " +
                                  $"Drafts: {turn.DraftTokensAccepted}/{turn.DraftTokensProposed} ({turn.AcceptanceRate * 100:F1}%)");
                Console.WriteLine($"  Input : {turn.Prompt}");
                Console.WriteLine($"  Output: {turn.Output}");
                if (turn.TurnNumber < turns.Count)
                    Console.WriteLine(subDivider);
            }

            Console.WriteLine(divider);
            Console.WriteLine();
        }
    }
}
using Spectre.Console;

namespace LLama.Examples;

internal static class UserSettings
{
    private static readonly string SettingsModelPath = Path.Join(AppContext.BaseDirectory, "DefaultModel.env");
    private static readonly string SettingsSpecTargetModelPath = Path.Join(AppContext.BaseDirectory, "DefaultSpecTargetModel.env");
    private static readonly string SettingsSpecDraftModelPath = Path.Join(AppContext.BaseDirectory, "DefaultSpecDraftModel.env");
    private static readonly string SettingsEmbedModelPath = Path.Join(AppContext.BaseDirectory, "DefaultEmbedModel.env");
    private static readonly string SettingsMMprojPath = Path.Join(AppContext.BaseDirectory, "DefaultMMProj.env");
    private static readonly string SettingsImagePath = Path.Join(AppContext.BaseDirectory, "DefaultImage.env");
    private static readonly string WhisperModelPath = Path.Join(AppContext.BaseDirectory, "DefaultWhisper.env");

    private static string? ReadDefaultPath(string file)
    {
        if (!File.Exists(file))
            return null;

        string path = File.ReadAllText(file).Trim();
        if (!File.Exists(path))
            return null;

        return path;
    }

    public static string GetEmbedModelPath(bool alwaysPrompt = false)
    {
        return PromptPath("embedmodel.gguf", SettingsEmbedModelPath, alwaysPrompt);
    }

    public static string GetModelPath(bool alwaysPrompt = false)
    {
        return PromptPath("model.gguf", SettingsModelPath, alwaysPrompt);
    }

    /// <summary>
    /// Prompts for the primary (large) target model path used by the speculative decoding benchmark.
    /// </summary>
    public static string GetSpeculativeTargetModelPath(bool alwaysPrompt = false)
    {
        return PromptPath("TARGET model.gguf (large primary model)", SettingsSpecTargetModelPath, alwaysPrompt);
    }

    /// <summary>
    /// Prompts for the secondary draft or split-MTP head model path used by the speculative decoding benchmark.
    /// </summary>
    public static string GetSpeculativeDraftModelPath(string label = "DRAFT model.gguf (smaller same-vocab model)", bool alwaysPrompt = false)
    {
        return PromptPath(label, SettingsSpecDraftModelPath, alwaysPrompt);
    }

    public static string GetMMProjPath(bool alwaysPrompt = false)
    {
        return PromptPath("mmproj", SettingsMMprojPath, alwaysPrompt);
    }    

    public static string GetImagePath(bool alwaysPrompt = false)
    {
        return PromptPath("image", SettingsImagePath, alwaysPrompt);
    }

    public static string GetWhisperPath(bool alwaysPrompt = false)
    {
        return PromptPath("whisper model.bin", WhisperModelPath, alwaysPrompt);
    }

    private static string PromptPath(string label, string saveFile, bool alwaysPrompt)
    {
        var defaultPath = ReadDefaultPath(saveFile);
        var rawPath = defaultPath is null || alwaysPrompt
            ? PromptUserForPath(label)
            : PromptUserForPathWithDefault(defaultPath, label);

        var path = NormalizePath(rawPath);

        if (File.Exists(path))
            WriteDefaultPath(saveFile, path);

        return path;

        static string NormalizePath(string input) => input.Trim().Trim('"');

        static void WriteDefaultPath(string settings, string path)
        {
            File.WriteAllText(settings, path);
        }

        static string PromptUserForPath(string text = "model")
        {
            return AnsiConsole.Prompt(
                new TextPrompt<string>($"Please input your {text} path:")
                   .PromptStyle("white")
                   .Validate(p => File.Exists(NormalizePath(p)), $"[red]ERROR: invalid {text} file path - file does not exist[/]")
            );
        }

        static string PromptUserForPathWithDefault(string defaultPath, string text = "model")
        {
            return AnsiConsole.Prompt(
                new TextPrompt<string>($"Please input your {text} path (or ENTER for default):")
                   .DefaultValue(defaultPath)
                   .PromptStyle("white")
                   .Validate(p => File.Exists(NormalizePath(p)), $"[red]ERROR: invalid {text} file path - file does not exist[/]")
            );
        }
    }
}

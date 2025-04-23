using System;
using LLama.Native;

namespace LLama.Exceptions;

/// <summary>
/// Base class for LLamaSharp runtime errors (i.e. errors produced by llama.cpp, converted into exceptions)
/// </summary>
public class RuntimeError
    : Exception
{
    /// <summary>
    /// Create a new RuntimeError
    /// </summary>
    /// <param name="message"></param>
    public RuntimeError(string message)
        : base(message)
    {

    }
}

/// <summary>
/// Loading model weights failed
/// </summary>
public class LoadWeightsFailedException
    : RuntimeError
{
    /// <summary>
    /// The model path which failed to load
    /// </summary>
    public string ModelPath { get; }

    /// <inheritdoc />
    public LoadWeightsFailedException(string modelPath)
        : base($"Failed to load model '{modelPath}'")
    {
        ModelPath = modelPath;
    }
}

/// <summary>
/// `llama_decode` return a non-zero status code
/// </summary>
public class LLamaDecodeError
    : RuntimeError
{
    /// <summary>
    /// The return status code
    /// </summary>
    public DecodeResult ReturnCode { get; }

    /// <inheritdoc />
    public LLamaDecodeError(DecodeResult returnCode)
        : base($"llama_decode failed: '{returnCode}'")
    {
        ReturnCode = returnCode;
    }
}

/// <summary>
/// `llama_decode` return a non-zero status code
/// </summary>
public class MissingTemplateException
    : RuntimeError
{
    /// <inheritdoc />
    public MissingTemplateException()
        : base("llama_chat_apply_template failed: template not found")
    {
    }
    
    /// <inheritdoc />
    public MissingTemplateException(string message)
        : base($"llama_chat_apply_template failed: template not found for '{message}'")
    {
    } 
}

/// <summary>
/// `llama_decode` return a non-zero status code
/// </summary>
public class TemplateNotFoundException
    : RuntimeError
{
    /// <inheritdoc />
    public TemplateNotFoundException(string name)
        : base($"llama_model_chat_template failed: Tried to retrieve template '{name}' but it couldn't be found.\n" +
                                            $"This might mean that the model was exported incorrectly, or that this is a base model that contains no templates.\n" +
                                            $"This exception can be disabled by passing 'strict=false' as a parameter when retrieving the template.")
    {
    } 
}

/// <summary>
/// `llama_get_logits_ith` returned null, indicating that the index was invalid
/// </summary>
public class GetLogitsInvalidIndexException
    : RuntimeError
{
    /// <summary>
    /// The incorrect index passed to the `llama_get_logits_ith` call
    /// </summary>
    public int Index { get; }

    /// <inheritdoc />
    public GetLogitsInvalidIndexException(int index)
        : base($"llama_get_logits_ith({index}) returned null")
    {
        Index = index;
    }
}
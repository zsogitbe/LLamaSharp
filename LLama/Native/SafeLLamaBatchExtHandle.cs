using System;
using Microsoft.Win32.SafeHandles;
using LLama.Exceptions;

namespace LLama.Native
{
    /// <summary>
    /// Safe handle wrapping a native <c>llama_batch_ext</c> pointer allocated via <see cref="NativeApi.llama_batch_ext_init"/>.
    /// </summary>
    public sealed class SafeLLamaBatchExtHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private SafeLLamaBatchExtHandle(IntPtr handle) : base(true)
        {
            SetHandle(handle);
        }

        /// <summary>
        /// Allocates a new extended batch associated with the specified native context.
        /// </summary>
        /// <param name="context">The context the extended batch will be processed against.</param>
        /// <returns>An owned <see cref="SafeLLamaBatchExtHandle"/>.</returns>
        /// <exception cref="RuntimeError">Thrown if native allocation returns <see cref="IntPtr.Zero"/>.</exception>
        public static SafeLLamaBatchExtHandle Create(SafeLLamaContextHandle context)
        {
            if (context is null || context.IsInvalid || context.IsClosed)
                throw new ArgumentException("Context handle must be valid and open.", nameof(context));

            IntPtr ptr = NativeApi.llama_batch_ext_init(context.DangerousGetHandle());
            if (ptr == IntPtr.Zero)
                throw new RuntimeError("Failed to initialize native llama_batch_ext.");

            return new SafeLLamaBatchExtHandle(ptr);
        }

        /// <summary>
        /// Clears all entries in this extended batch for reuse.
        /// </summary>
        public void Clear()
        {
            if (IsInvalid || IsClosed)
                throw new ObjectDisposedException(nameof(SafeLLamaBatchExtHandle));

            NativeApi.llama_batch_ext_clear(handle);
        }

        /// <inheritdoc />
        protected override bool ReleaseHandle()
        {
            NativeApi.llama_batch_ext_free(handle);
            return true;
        }
    }
}
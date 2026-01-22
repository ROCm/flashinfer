"""AMD FlashInfer JIT Cache Package

This package provides pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.
"""

try:
    from ._build_meta import __version__
except ImportError:
    # Fallback for development or when _build_meta.py hasn't been generated yet
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            __version__ = version("amd-flashinfer-jit-cache")
        except PackageNotFoundError:
            __version__ = "0.0.0+unknown"
    except ImportError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

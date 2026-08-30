"""Make pip-installed NVIDIA CUDA libraries visible to native loaders."""

from __future__ import annotations

import glob
import os


def setup_nvidia_lib_path() -> None:
    """Add pip-installed NVIDIA lib dirs to LD_LIBRARY_PATH (idempotent).

    ctranslate2 (faster-whisper) and torchcodec (torchaudio's save path,
    which demucs uses to write its stems) dlopen CUDA shared libs (cublas,
    cudnn, npp, …) at runtime but don't find them inside pip's nvidia-*
    packages automatically. The variable is inherited by subprocesses, so
    demucs' `python -m demucs` child sees them too.
    """
    try:
        site_packages = os.path.dirname(
            os.path.dirname(__import__("nvidia.cublas", fromlist=["lib"]).__file__)
        )
        lib_dirs = glob.glob(os.path.join(site_packages, "*/lib"))
        if not lib_dirs:
            return
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        present = set(existing.split(":")) if existing else set()
        missing = [d for d in lib_dirs if d not in present]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(missing) + (
                ":" + existing if existing else ""
            )
    except (ImportError, AttributeError):
        pass

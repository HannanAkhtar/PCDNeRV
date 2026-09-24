"""Runtime validation shared by compute-pilot entry points."""

from __future__ import annotations

import json

import torch


def cuda_environment():
    available = bool(torch.cuda.is_available())
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
    }


def validate_cuda_environment(*, require_cuda=True, report=True):
    """Report the installed PyTorch/CUDA pairing and optionally require CUDA."""
    environment = cuda_environment()
    if report:
        print(json.dumps({"runtime_environment": environment}, indent=2))
    if require_cuda and not environment["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable in the installed PyTorch environment. "
            "On Colab, select a GPU runtime and use its preinstalled PyTorch; "
            "do not attempt to repair this by pip-installing another torch wheel."
        )
    return environment

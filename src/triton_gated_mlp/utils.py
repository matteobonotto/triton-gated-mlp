import torch
from torch import Tensor, nn
import os

from pathlib import Path
import stat
from .const import ROOT_PATH


### triton-dejavu cache data
def setup_dejavu_cache_dir():
    path = Path("triton_dejavu_cache")

    p = Path(path)

    # 1. Create directory if it does not exist
    p.mkdir(parents=True, exist_ok=True)

    # 2. Ensure permissions: o+rw (add read & write for others)
    st = p.stat()
    new_mode = st.st_mode | stat.S_IROTH | stat.S_IWOTH
    os.chmod(p, new_mode)

    os.environ["TRITON_DEJAVU_STORAGE"] = str(ROOT_PATH / path)


def copy_weights(model_src: nn.Module, model_tgt: nn.Module) -> None:
    """Copy weights of model_src into model_tgt"""
    for src, tgt in zip(model_src.named_parameters(), model_tgt.named_parameters()):
        name_src, param_src = src
        name_tgt, param_tgt = tgt
        if name_src == name_tgt:
            param_tgt.data = param_src.data.clone()


def validate_contiguous(x: Tensor) -> Tensor:
    return x if x.is_contiguous() else x.contiguous()


def validate_tensor_device(x: Tensor):
    if not x.is_cuda:
        message = "Tensor must be on CUDA or TRITON_INTERPRET must be set to '1'"
        assert os.environ.get("TRITON_INTERPRET", False) == "1", message


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    device = "cuda" if is_cuda() else "cpu"
    return torch.device(device)


def is_cuda():
    return torch.cuda.is_available()

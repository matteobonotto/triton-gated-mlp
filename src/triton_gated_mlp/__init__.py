from pathlib import Path
import os

from . import utils, autotune, const, gated_mlp

# utils.setup_dejavu_cache_dir()

if not utils.is_cuda_available() or "TRITON_IS_DEBUGGING" in os.environ.keys():
    os.environ["TRITON_INTERPRET"] = "1"

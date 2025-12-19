import os
from .utils import is_cuda_available
from pathlib import Path
from .const import ROOT_PATH

import stat


### triton-dejavu cache data
def ensure_dejavu_cache_dir(path: Path):
    p = Path(path)

    # 1. Create directory if it does not exist
    p.mkdir(parents=True, exist_ok=True)

    # 2. Ensure permissions: o+rw (add read & write for others)
    st = p.stat()
    new_mode = st.st_mode | stat.S_IROTH | stat.S_IWOTH
    os.chmod(p, new_mode)


dir = Path("triton_dejavu_cache")
ensure_dejavu_cache_dir(path = dir)

os.environ["TRITON_DEJAVU_STORAGE"] = str(ROOT_PATH / dir)

if not is_cuda_available() or "TRITON_IS_DEBUGGING" in os.environ.keys():
    os.environ["TRITON_INTERPRET"] = "1"

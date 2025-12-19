import numpy as np
import torch
from tqdm import tqdm

from triton_gated_mlp.gated_mlp import FusedGatedMLP
from triton_gated_mlp.utils import get_device, setup_dejavu_cache_dir

DTYPE = torch.float16
DEVICE = get_device()


def get_gated_mlp(N, K):
    gmlp_kwargs = {
        "hidden_size": K,
        "intermediate_size": N,
        "dropout_p": 0.0,
        "bias": False,
    }
    return FusedGatedMLP(**gmlp_kwargs).to(DEVICE).to(DTYPE)


def step(mlp, x):
    out = mlp(x)
    loss = out.sum()
    loss.backward()


if __name__ == "__main__":
    setup_dejavu_cache_dir()
    Ms = [int(2**i) for i in np.arange(5, 8, 1)]

    ### Run autotune using triton-dejavu
    for M in tqdm(Ms, total=len(Ms)):
        print(" *** Autotuning for M = N = K ***")
        N = K = M
        gmlp = get_gated_mlp(N=N, K=K)

        print(f"Tuning for: {DTYPE=}, {M=}, {N=}, {K=}")

        x = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        x.requires_grad = True
        step(gmlp, x)

    ### Run autotune using triton-dejavu
    for M in tqdm(Ms, total=len(Ms)):
        M *= 4
        K *= 2

        print(" *** Autotuning for M = 4*N = 2*K ***")
        N = K = M
        gmlp = get_gated_mlp(N=N, K=K)

        print(f"Tuning for: {DTYPE=}, {M=}, {N=}, {K=}")

        x = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        x.requires_grad = True
        step(gmlp, x)

import numpy as np
import torch
from tqdm import tqdm

from .gated_mlp import FusedGatedMLP
from .utils import get_device

DTYPE = torch.float16
DEVICE = get_device()


def get_gated_mlp():
    kwargs = {"device": DEVICE, "dtype": DTYPE}

    gmlp_kwargs = {
        "hidden_size" : K, 
        "intermediate_size" : N,
        "dropout_p" : 0.0, 
        "bias" : False,
    }
    return FusedGatedMLP(**gmlp_kwargs).to(DEVICE).to(DTYPE)


def step(mlp, x):
    out = mlp(x)
    loss = out.sum()
    loss.backward()


if __name__ == "__main__":
    ### Run autotune using triton-dejavu
    Ms = [int(2 ** i) for i in np.arange(5, 13, 1)]
    for M in tqdm(Ms, total=len(M)):
        N = K = M
        gmlp = get_gated_mlp(M, N, K)

        print(f"Tuning for: {DTYPE=}, {M=}, {N=}, {K=}")
    
        x = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        x.requires_grad = True
        step(gmlp, x)



import os

os.environ["TRITON_INTERPRET"] = "1"

import torch

from triton_gated_mlp.ops.fwd_block import mlp_hidden_states_fwd

DEVICE = torch.device("cuda")


def tiled(x, Wu, Wg, act, BLOCK_SIZE_M, BLOCK_SIZE_N):
    xo_ = torch.zeros_like(x)
    M, K = x.shape
    N, _ = Wu.shape

    for n in range(0, N, BLOCK_SIZE_N):
        for m in range(0, M, BLOCK_SIZE_M):
            tile_x = x[m : m + BLOCK_SIZE_M, :]
            tile_Wu = Wu[n : n + BLOCK_SIZE_N, :]
            tile_Wg = Wg[n : n + BLOCK_SIZE_N, :]

            tile_xp = (tile_x @ tile_Wu.T) * act(tile_x @ tile_Wg.T)

            xo_[m : m + BLOCK_SIZE_M, n : n + BLOCK_SIZE_N] = tile_xp
            ...

    return xo_





def test_fusing_more():

    torch.manual_seed(42)

    (
        M,
        N,
        K,
    ) = (
        6,
        6,
        6,
    )

    # x = torch.rand(S, D1)
    # WuT = torch.rand(D1, D2)
    # WgT = torch.rand(D1, D2)
    # WoT = torch.rand(D2, D1)

    x = torch.rand(M, K).to(DEVICE) 
    Wu = torch.rand(N, K).to(DEVICE)
    Wg = torch.rand(N, K).to(DEVICE)
    Wo = torch.rand(K, N).to(DEVICE)

    act = torch.nn.functional.silu
    # act = lambda x: x

    xp = (x @ Wu.T) * act(x @ Wg.T)
    # xo = xp @ Wo.T

    ### do it in a tiled way: outer loop along Wo rows, inner loop on x rows and (Wu, Wg) cols
    BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 4
    xp_ = tiled(x, Wu, Wg, act, BLOCK_SIZE_M, BLOCK_SIZE_N)
    assert ((xp - xp_).norm() / xp.norm()) < 1e-6
    print(xp_)

    # x =

    xp__ = mlp_hidden_states_fwd(x, Wu=Wu, Wg=Wg, act_fn="silu")
    assert ((xp - xp__).norm() / xp.norm()) < 1e-6


test_fusing_more()

import os

os.environ["TRITON_INTERPRET"] = "1"

import torch
from triton_gated_mlp.ops.fwd_fusing_more import mlp_hidden_states_fwd

DEVICE = torch.device("cuda")


def all_fused_algo(x, WoT, WuT, WgT, act, BLOCK_SIZE_M, BLOCK_SIZE_N):
    xo_ = torch.zeros_like(x)
    M, K = x.shape
    _, N = WgT.shape

    for n in range(0, N, BLOCK_SIZE_N):
        for m in range(0, M, BLOCK_SIZE_M):
            tile_x = x[m : m + BLOCK_SIZE_M, :]
            tile_WoT = WoT[n : n + BLOCK_SIZE_N, :]
            tile_WuT = WuT[:, n : n + BLOCK_SIZE_N]
            tile_WgT = WgT[:, n : n + BLOCK_SIZE_N]

            tile_xp = (tile_x @ tile_WuT) * act(tile_x @ tile_WgT)
            tile_xo_partial = tile_xp @ tile_WoT

            xo_[m : m + BLOCK_SIZE_M, :] += tile_xo_partial
            ...

    return xo_


def test_fusing_more():

    torch.manual_seed(42)

    (
        M,
        K,
        N,
    ) = (
        16,
        16,
        16,
    )

    # x = torch.rand(S, D1)
    # WuT = torch.rand(D1, D2)
    # WgT = torch.rand(D1, D2)
    # WoT = torch.rand(D2, D1)

    x = torch.rand(M, K).to(DEVICE) / 10
    Wu = torch.rand(N, K).to(DEVICE)
    Wg = torch.rand(N, K).to(DEVICE)
    Wo = torch.rand(K, N).to(DEVICE)

    act = torch.nn.functional.silu
    # act = lambda x: x

    xp = (x @ Wu.T) * act(x @ Wg.T)
    xo = xp @ Wo.T

    ### do it in a tiled way: outer loop along Wo rows, inner loop on x rows and (Wu, Wg) cols
    BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 4
    xo_ = all_fused_algo(x, Wo.T, Wu.T, Wg.T, act, BLOCK_SIZE_M, BLOCK_SIZE_N)
    assert ((xo - xo_).norm() / xo.norm()) < 1e-6
    print(xo_)

    # x =

    xo__ = mlp_hidden_states_fwd(x, WT_up=Wu, WT_gp=Wg, WT_op=Wo, act_fn="silu")
    assert ((xo - xo__).norm() / xo.norm()) < 1e-6


test_fusing_more()

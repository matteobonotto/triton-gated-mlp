import os

os.environ["TRITON_INTERPRET"] = "1"

import pytest
import random
import torch
from triton_gated_mlp.ops.fwd_block import mlp_hidden_states_fwd

DEVICE = torch.device("cuda")


def tiled(x, Wu, Wg, bu, bg, act, BLOCK_SIZE_M, BLOCK_SIZE_N):
    M, K = x.shape
    N, _ = Wu.shape
    xo_ = torch.zeros(M, N).to(x.device)

    for n in range(0, N, BLOCK_SIZE_N):
        for m in range(0, M, BLOCK_SIZE_M):
            tile_x = x[m : m + BLOCK_SIZE_M, :]
            tile_Wu = Wu[n : n + BLOCK_SIZE_N, :]
            tile_Wg = Wg[n : n + BLOCK_SIZE_N, :]

            tile_bu = bu[n : n + BLOCK_SIZE_N]
            tile_bg = bg[n : n + BLOCK_SIZE_N]

            tile_xp = (tile_x @ tile_Wu.T + tile_bu[None, :]) * act(
                tile_x @ tile_Wg.T + tile_bg[None, :]
            )

            xo_[m : m + BLOCK_SIZE_M, n : n + BLOCK_SIZE_N] = tile_xp
            ...

    return xo_


@pytest.mark.skip("dev")
def test_dev():

    torch.manual_seed(42)

    (
        M,
        N,
        K,
    ) = (
        80,
        89,
        37,
    )

    x = torch.rand(M, K).to(DEVICE)
    Wu = torch.rand(N, K).to(DEVICE)
    bu = torch.rand(N).to(DEVICE)
    Wg = torch.rand(N, K).to(DEVICE)
    bg = torch.rand(N).to(DEVICE)
    Wo = torch.rand(K, N).to(DEVICE)

    from torch import nn

    l = nn.Linear(K, N).to(DEVICE)

    l(x)

    p = 0
    # x = create_tensor(M, K).to(DEVICE)
    # print(x)
    # Wu = create_tensor(N, K).to(DEVICE)
    # Wg = create_tensor(N, K).to(DEVICE)
    # Wo = create_tensor(K, N).to(DEVICE)

    act = torch.nn.functional.silu
    # act = lambda x: x

    xp = torch.nn.functional.dropout((x @ Wu.T + bu) * act(x @ Wg.T + bg), p=p)
    # xo = xp @ Wo.T

    ### do it in a tiled way: outer loop along Wo rows, inner loop on x rows and (Wu, Wg) cols
    BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 4
    xp_ = tiled(x, Wu, Wg, bu, bg, act, BLOCK_SIZE_M, BLOCK_SIZE_N)
    if p == 0.0:
        assert ((xp - xp_).norm() / xp.norm()) < 1e-6
    # print(xp_)

    # x =

    xp__, mask = mlp_hidden_states_fwd(
        x, Wu=Wu, Wg=Wg, bu=bu, bg=bg, act_fn="silu", dropout_p=p
    )
    assert ((xp - xp__).norm() / xp.norm()) < 1e-6

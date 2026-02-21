import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

import pytest
import torch
import random

from triton_gated_mlp.ops.fwd_block import mlp_hidden_states_fwd

from utils import *

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


def test_no_bias():
    for _ in range(3):
        (
            x,
            Wu,
            bu,
            Wg,
            bg,
        ) = create_tensors(
            M=random.randint(16, 64),
            N=random.randint(16, 64),
            K=random.randint(16, 64),
            DEVICE=DEVICE,
        )

        # (
        #     x,
        #     Wu,
        #     bu,
        #     Wg,
        #     bg,
        # ) = create_tensors(8, 8, 8,DEVICE=DEVICE,
        # )

        act = torch.nn.functional.silu
        xp = torch.nn.functional.dropout((x @ Wu.T) * act(x @ Wg.T), p=0.0)
        xp__, _ = mlp_hidden_states_fwd(
            x, Wu=Wu, Wg=Wg, bu=None, bg=None, act_fn="silu", dropout_p=0.0
        )
        assert ((xp - xp__).norm() / xp.norm()) < 1e-6


def test_bias():
    for _ in range(3):
        (
            x,
            Wu,
            bu,
            Wg,
            bg,
        ) = create_tensors(
            M=random.randint(16, 64),
            N=random.randint(16, 64),
            K=random.randint(16, 64),
            DEVICE=DEVICE,
        )

        # (
        #     x,
        #     Wu,
        #     bu,
        #     Wg,
        #     bg,
        # ) = create_tensors(8, 8, 8,DEVICE=DEVICE,
        # )

        act = torch.nn.functional.silu
        xu = x @ Wu.T + bu
        xg = act(x @ Wg.T + bg)
        xp = torch.nn.functional.dropout(xu * xg, p=0.0)
        # print(f"{xu=} \n{xg=} \n{xp=} \n")
        xp__, _ = mlp_hidden_states_fwd(
            x, Wu=Wu, Wg=Wg, bu=bu, bg=bg, act_fn="silu", dropout_p=0.0
        )
        assert ((xp - xp__).norm() / xp.norm()) < 1e-6


# test_bias()


def test_droput():
    p = 0.1
    for _ in range(3):
        (
            x,
            Wu,
            bu,
            Wg,
            bg,
        ) = create_tensors(
            M=random.randint(16, 64),
            N=random.randint(16, 64),
            K=random.randint(16, 64),
            DEVICE=DEVICE,
        )

        # (
        #     x,
        #     Wu,
        #     bu,
        #     Wg,
        #     bg,
        # ) = create_tensors(8, 8, 8,DEVICE=DEVICE,
        # )

        act = torch.nn.functional.silu
        xp = torch.nn.functional.dropout((x @ Wu.T + bu) * act(x @ Wg.T + bg), p=p)
        xp__, _ = mlp_hidden_states_fwd(
            x, Wu=Wu, Wg=Wg, bu=bu, bg=bg, act_fn="silu", dropout_p=p
        )
        ...
        # assert ((xp - xp__).norm() / xp.norm()) < 1e-6


# test_droput()

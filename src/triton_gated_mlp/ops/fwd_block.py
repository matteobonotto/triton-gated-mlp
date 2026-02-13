import torch
from torch import Tensor
import triton
import triton.language as tl
from typing import Optional

from .utils import (
    validate_dimensions_gmlp,
    get_num_streaming_multiprocessors,
    get_target_dtype,
    cdiv,
    map_pid_m_n,
)
from .activations import _act_fwd


def is_at_least_hopper() -> bool:
    return torch.cuda.get_device_capability()[0] >= 9


def maybe_flatten(warp_specialize: bool) -> bool:
    return False if (warp_specialize and is_at_least_hopper()) else True


"""
        _fwd_kernel[grid](
            x,
            Wu,
            Wg,
            xo,
            x.stride(0),
            x.stride(1),
            Wu.stride(0),
            Wu.stride(1),
            Wg.stride(0),
            Wg.stride(1),
            xo.stride(0),
            xo.stride(1),
            act_fn,
            dropout_p,
            M,
            N,
            K,
            NUM_SMS,
            flatten, 
            warp_specialize,
            target_dtype, 
        )
    """


@triton.jit()
def _dropout_fwd(
    x_ptr,
):
    raise NotImplementedError("_dropout_fwd not implemented yet!!")


def autotune_config(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": GS,
            },
            num_stages=s,
            num_warps=w,
        )  #
        for BM in [
            32,
            64,
            128,
        ]  #
        for BN in [
            64,
            128,
            256,
        ]  #
        for BK in [16, 32, 64]  #
        for GS in [2, 4]  #
        for s in ([1, 2, 4])  #
        for w in [4, 8]  #
    ]


# @triton.autotune(
#     configs=autotune_config(),
#     key=["M", "N", "K"],
#     # prune_configs_by={'early_config_prune': prune_invalid_configs}
# )
@triton.jit()
def _fwd_kernel(
    x_ptr,
    Wu_ptr,
    Wg_ptr,
    xo_ptr,
    x_str_0,
    x_str_1,
    Wu_str_0,
    Wu_str_1,
    Wg_str_0,
    Wg_str_1,
    xo_str_0,
    xo_str_1,
    act_fn: tl.constexpr,
    dropout_p,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS,
    flatten: tl.constexpr,
    warp_specialize: tl.constexpr,
    target_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    ...

    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs = num_programs_n * num_programs_m
    optimize_L2 = True

    arange_M = tl.arange(0, BLOCK_SIZE_M)
    arange_K = tl.arange(0, BLOCK_SIZE_K)
    arange_N = tl.arange(0, BLOCK_SIZE_N)

    for pid_ in tl.range(
        pid,
        num_programs,
        step=NUM_SMS,
        flatten=flatten,
        warp_specialize=warp_specialize,
    ):
        pid_m, pid_n = map_pid_m_n(
            pid_, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
        )

        offset_m = pid_m * BLOCK_SIZE_M + arange_M
        offset_n = pid_n * BLOCK_SIZE_N + arange_N

        tile_x_ptr = x_ptr + offset_m[:, None] * x_str_0 + arange_K[None, :] * x_str_1
        tile_Wu_ptr = (
            Wu_ptr + offset_n[:, None] * Wu_str_0 + arange_K[None, :] * Wu_str_1
        )
        tile_Wg_ptr = (
            Wg_ptr + offset_n[:, None] * Wg_str_0 + arange_K[None, :] * Wg_str_1
        )

        tile_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_SIZE_K):
            offset_k = k + arange_K
            mask_k = offset_k < K

            tile_x = tl.load(tile_x_ptr, mask_k, other=0.0)
            tile_Wg = tl.load(tile_Wg_ptr, mask_k, other=0.0)
            tile_Wu = tl.load(tile_Wu_ptr, mask_k, other=0.0)

            tile_u = tl.dot(tile_x, tile_Wu.T, acc=tile_u)
            tile_g = tl.dot(tile_x, tile_Wg.T, acc=tile_g)
            ...

            # TODO: slide the tile_a, tile_b pointers along Z direction of BLOCK_SIZE_K steps

        # multiply and apply activation
        tile_o = tile_u * _act_fwd(tile_g, act_fn)

        # apply dropout if needed
        if dropout_p > 0:
            tile_o = _dropout_fwd(tile_o)

        ...

        ###
        tile_xo = xo_ptr + offset_m[:, None] * xo_str_0 + offset_n[None, :] * xo_str_1
        mask_o = (offset_m < M)[:, None] & (offset_n < N)[None, :]
        tl.store(tile_xo, value=tile_o, mask=mask_o)



@torch.no_grad()
def mlp_hidden_states_fwd(
    x: Tensor,
    Wu: Tensor,  # this one is transposed
    Wg: Tensor,  # this one is transposed
    act_fn: str,
    bg: Tensor | None = None,
    bu: Tensor | None = None,
    dropout_p: float = 0.0,
    warp_specialize: bool = False,
) -> Tensor:
    """
    This function computes the follwing operations in a fused fashion:

        self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

    Wieghts WT_up and WT_gp are kept transposed, and they are transposed back inside the
    triton kernel when doing tl.dot(x, W.T, ...)
    """

    ### validate input dimension
    validate_dimensions_gmlp(hidden_states=x, Wu=Wu, Wg=Wg, bu=bu, bg=bg)

    M, K = x.shape
    N, _ = Wu.shape

    ### triton tensor_descriptor needs tensors to have stride(0) that
    # is a multiple of 16. Check this and pad if needed.
    # if K % 16 != 0:
    #     raise NotImplementedError()
    #     # a = pad_tensor_16_byte_aligned(a, axis=1)
    #     # b = pad_tensor_16_byte_aligned(b, axis=0)

    # if N % 16 != 0:
    #     raise NotImplementedError()
    #     # b = pad_tensor_16_byte_aligned(b, axis=1)
    #     # old_N = N

    ### Create the grid: we are using tensor_descriptor in a persistent
    # implementation (one kernel may execute more programs of the grid,
    # not just a single one).
    NUM_SMS = get_num_streaming_multiprocessors()
    # grid = lambda META: (min(
    # (NUM_SMS, cdiv(K, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]))
    # ),)
    flatten = maybe_flatten(warp_specialize)
    (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    ) = (
        4,
        4,
        4,
        1,
    )
    grid = (min((NUM_SMS, cdiv(K, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N))),)
    ###
    # xo = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    xo = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    target_dtype = get_target_dtype(x)

    with torch.cuda.device(x.device):
        _fwd_kernel[grid](
            x,
            Wu,
            Wg,
            xo,
            x.stride(0),
            x.stride(1),
            Wu.stride(0),
            Wu.stride(1),
            Wg.stride(0),
            Wg.stride(1),
            xo.stride(0),
            xo.stride(1),
            act_fn,
            dropout_p,
            M,
            N,
            K,
            NUM_SMS,
            flatten,
            warp_specialize,
            target_dtype,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
        )

    return xo.to(x.dtype) if xo.dtype != x.dtype else xo

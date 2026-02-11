import triton
import triton.language as tl
import torch
from typing import Optional
import math
from torch import Tensor

from .utils import (
    get_num_streaming_multiprocessors,
    validate_dimensions,
    cdiv,
    map_pid_m_n,
)
from .fwd import _act_fwd


@triton.jit()
def _fed_kernel(
    x_ptr,
    WT_up_ptr,
    WT_gp_ptr,
    WT_op_ptr,
    x_out_ptr,
    act_fn: tl.constexpr,
    dropout_p,
    M,
    N,
    K,
    NUM_SMS,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs = num_programs_m * num_programs_n
    optimize_L2 = True

    ### create tensor_descriptor(s) for all the involved tensors
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    WT_up_desc = tl.make_tensor_descriptor(
        WT_up_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    WT_gp_desc = tl.make_tensor_descriptor(
        WT_gp_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    WT_op_desc = tl.make_tensor_descriptor(
        WT_op_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    x_out_desc = tl.make_tensor_descriptor(
        x_out_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )

    ### persistent matmul: use the same kernel to compute several tiles of x_out
    for pid_ in tl.range(pid, num_programs, NUM_SMS):
        ### map pid to pid_m, pid_n
        pid_m, pid_n = map_pid_m_n(
            pid_, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

        print(pid, pid_, pid_m, pid_n, offset_m, offset_n)

        ### compute tile_p = (tile_x @ tile_WT_up) * act(tile_x @ tile_WT_gp)
        tile_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_gp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for offset_k in tl.range(0, K, BLOCK_SIZE_K):
            tile_x = x_desc.load(offsets=[offset_m, offset_k])
            tile_WT_up = WT_up_desc.load(offsets=[offset_n, offset_k])
            tile_gp_up = WT_gp_desc.load(offsets=[offset_n, offset_k])

            tile_up = tl.dot(tile_x, tile_WT_up.T, acc=tile_up)
            tile_gp = tl.dot(tile_x, tile_gp_up.T, acc=tile_gp)

        tile_prod = tile_up * _act_fwd(tile_gp, act_fn)

        ### compute tile_o = tile_prod @ WT_op.T
        for offset_k in tl.range(0, K, BLOCK_SIZE_K):
            WT_op = WT_op_desc.load(offsets=[offset_k, offset_n])
            acc = x_out_desc.load(offsets=[offset_m, offset_k])
            out = tl.dot(tile_prod, WT_op.T, acc=acc)
            x_out_desc.store(offsets=[offset_m, offset_k], value=out)


@torch.no_grad()
def mlp_hidden_states_fwd(
    x: Tensor,
    WT_up: Tensor,  # this one is transposed
    WT_gp: Tensor,  # this one is transposed
    act_fn: str,
    WT_op: Tensor,  # this one is transposed
    b_gp: Tensor | None = None,
    b_op: Tensor | None = None,
    b_up: Tensor | None = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """
    This function computes the follwing operations in a fused fashion:

        self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

    Wieghts WT_up and WT_gp are kept transposed, and they are transposed back inside the
    triton kernel when doing tl.dot(x, W.T, ...)
    """

    ### validate input dimension
    validate_dimensions(x, WT_up, b_up, WT_gp, b_gp)

    M, K = x.shape
    N, _ = WT_up.shape

    ### triton tensor_descriptor needs tensors to have stride(0) that
    # is a multiple of 16. Check this and pad if needed.
    if K % 16 != 0:
        raise NotImplementedError()
        # a = pad_tensor_16_byte_aligned(a, axis=1)
        # b = pad_tensor_16_byte_aligned(b, axis=0)

    if N % 16 != 0:
        raise NotImplementedError()
        # b = pad_tensor_16_byte_aligned(b, axis=1)
        # old_N = N

    ### Create the grid: we are using tensor_descriptor in a persistent
    # implementation (one kernel may execute more programs of the grid,
    # not just a single one).
    def get_dummy_META():
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 8,
            "BLOCK_SIZE_K": 16,
            "GROUP_SIZE_M": 1,
        }

    META = get_dummy_META()
    NUM_SMS = get_num_streaming_multiprocessors()
    # NUM_SMS = 2
    grid = (
        min((NUM_SMS, cdiv(K, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]))),
    )
    # grid = lambda META: min(
    #     (NUM_SMS, cdiv(K, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]))
    # ),

    ### custom allocation function
    def allocator(size, stream: int, allignment: Optional[int]):
        return torch.empty(size, device=x.device, dtype=torch.int8)

    triton.set_allocator(allocator)

    ###
    x_out = torch.zeros_like(x, dtype=x.dtype, device=x.device)

    with torch.cuda.device(x.device):
        _fed_kernel[grid](
            x,
            WT_up,
            WT_gp,
            WT_op,
            x_out,
            act_fn,
            dropout_p,
            M,
            N,
            K,
            NUM_SMS,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            META["GROUP_SIZE_M"],
        )

    return x_out

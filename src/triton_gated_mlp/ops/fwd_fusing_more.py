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
    get_shared_mem_limit,
    is_hopper,
    map_dtype, 
    get_target_dtype,
)
from .fwd import _act_fwd



def prune_invalid_configs(configs, named_args, **kwargs):
    M = named_args["M"]; N = named_args["N"]; K = named_args["K"]

    # Hopper SMEM per-SM is 228KB physical, but many toolchains effectively cap at 99KB
    # Your error reports 101376 bytes, so use that.
    SMEM_LIMIT = get_shared_mem_limit(torch.device("cuda:0"))

    # infer element size (rough). If you always use fp16/bf16 loads for x/W, set 2.
    # If you sometimes stage fp32, set 4.
    BYTES = 2 #if DTYPE == torch.float32 else 2 

    pruned = []
    for cfg in configs:
        BM = cfg.kwargs["BLOCK_SIZE_M"]
        BN = cfg.kwargs["BLOCK_SIZE_N"]
        BK = cfg.kwargs["BLOCK_SIZE_K"]

        # basic "fits problem" (optional)
        if BM > M or BN > N or BK > K:
            continue

        num_stages = getattr(cfg, "num_stages", None)
        if num_stages is None:
            num_stages = 2  # conservative default if you don't set it per-config

        # Conservative estimate: tiles staged simultaneously per stage
        # x + up + gp + op  (you can tune this formula to your kernel’s actual staging)
        smem_per_stage = (
            BM * BK   # x
            + BN * BK # WT_up
            + BN * BK # WT_gp
            + BK * BN # WT_op
        ) * BYTES

        smem_est = smem_per_stage * num_stages

        if smem_est <= SMEM_LIMIT:
            pruned.append(cfg)

    if not pruned:
        pruned = [configs[0]]  # keep something
    return pruned


def autotune_config(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS, 
            }, num_stages=s, num_warps=w)  #
        for BM in [32, 64, 128,]  #
        for BN in [64, 128, 256, ]  #
        for BK in [16, 32, 64]  #
        for GS in [2, 4]  #
        for s in ([1, 2, 4])  #
        for w in [4, 8]  #
    ]

@triton.autotune(
    configs=autotune_config(), 
    key=["M", "N", "K"],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
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
    for pid_ in tl.range(pid, num_programs, NUM_SMS, flatten=flatten, warp_specialize=warp_specialize):
        ### map pid to pid_m, pid_n
        pid_m, pid_n = map_pid_m_n(
            pid_, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

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
            WT_op = WT_op_desc.load(offsets=[offset_k, offset_n])#.to(tl.float32)
            acc = x_out_desc.load(offsets=[offset_m, offset_k]).to(tl.float32)
            acc = tl.dot(tile_prod, WT_op.T, acc=acc)
            x_out_desc.store(offsets=[offset_m, offset_k], value=map_dtype(acc, target_dtype))



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
    warp_specialize: bool = False, 
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
    NUM_SMS = get_num_streaming_multiprocessors()
    grid = lambda META: (min(
        (NUM_SMS, cdiv(K, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]))
    ),)
    flatten = False if (warp_specialize and is_hopper()) else True

    ### custom allocation function
    def allocator(size, stream: int, allignment: Optional[int]):
        return torch.empty(size, device=x.device, dtype=torch.int8)
    triton.set_allocator(allocator)

    ###
    # x_out = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    x_out = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    target_dtype = get_target_dtype(x)

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
            flatten, 
            warp_specialize,
            target_dtype, 
        )

    return x_out.to(x.dtype) if x_out.dtype != x.dtype else x_out

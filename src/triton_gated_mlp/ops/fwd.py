from typing import Optional
import triton
import triton.language as tl
import torch
from torch import Tensor
import math

from pathlib import Path
import os

# if os.environ.get("storage_prefix")

# if "triton_dejavu_cache" in Path()

from .utils import validate_dimensions
from ..const import ROOT_PATH

CACHE_DIR = ROOT_PATH / Path("triton_dejavu_cache")
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )


import triton_dejavu

from .act import _act_fwd, _act_bwd
from .utils import map_pid_m_n, get_num_streaming_multiprocessors

def get_smem_limit(device=None):
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # This is the maximum dynamic shared memory per block (bytes)
    return props.shared_memory_per_block

def prune_invalid_configs(configs, named_args, **kwargs):
    M = named_args["M"]; N = named_args["N"]; K = named_args["K"]

    # Hopper SMEM per-SM is 228KB physical, but many toolchains effectively cap at 99KB
    # Your error reports 101376 bytes, so use that.
    SMEM_LIMIT = get_smem_limit(torch.device("cuda:0"))

    # infer element size (rough). If you always use fp16/bf16 loads for x/W, set 2.
    # If you sometimes stage fp32, set 4.
    BYTES = 2

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

def launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def get_autotune_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": GS,
            },
            pre_hook=pre_hook,
        )  #
        for BM in [32, 64, 128, 256]  #
        for BN in [32, 64, 128, 256]  #
        for BK in [32, 64, 128, 256]  #
        for GS in [2, 4, 8, 16]  #
        # for s in ([2])  #
        # for w in [4]  #
        # for SUBTILE in [True, False]  #
    ]


def get_autotune_config_space():
    BM = [16, 32, 64, 128]  #
    BN = [16, 32, 64, 128]  #
    BK = [16, 32, 64, 128]  #
    GS = [2, 4, 8, 16]
    return triton_dejavu.ConfigSpace(
        {
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": GS,
        },
        # num_warps=[4, 8, 16],
        # num_stages=[1, 2, 4, 6],
    )


# @triton_dejavu.autotune(
#     # configs=get_autotune_configs(),
#     config_space=get_autotune_config_space(),
#     key=["M", "N", "K"],
#     use_bo=True,
# )
@triton.autotune(
    configs=get_autotune_configs(), 
    key=["M", "N", "K"],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit()
def _fwd_kernel(
    x_ptr,
    WT_up_ptr,
    # b_up_ptr,
    # has_bias_up,
    WT_gp_ptr,
    # b_gp_ptr,
    # has_bias_gp,
    out_ptr,
    act_fn: tl.constexpr,
    dropout_p,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs_k = tl.cdiv(K, BLOCK_SIZE_K)
    total_programs = num_programs_m * num_programs_n

    ### define tensor descriptors

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

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    # if has_bias_up:
    #     b_up_desc = tl.make_tensor_descriptor(
    #         b_up_ptr, shape=[N, 1], strides=[1, 0], block_shape=[BLOCK_SIZE_N]
    #     )

    # if has_bias_gp:
    #     b_gp_desc = tl.make_tensor_descriptor(
    #         b_gp_ptr, shape=[N, 1], strides=[1, 0], block_shape=[BLOCK_SIZE_N]
    #     )

    ### persistent matmul: loop over multiple (m,n) tiles
    for tile_id in tl.range(
        pid, total_programs, NUM_SMS, flatten=True, warp_specialize=False
    ):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, True
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

        tile_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_gp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(0, num_programs_k):

            offset_k = k * BLOCK_SIZE_K

            tile_x = x_desc.load([offset_m, offset_k])
            tile_WT_up = WT_up_desc.load([offset_n, offset_k])

            tile_up = tl.dot(tile_x, tile_WT_up.T, acc=tile_up)
            # if has_bias_up:
            #     tile_up += ...

            tile_WT_gp = WT_gp_desc.load([offset_n, offset_k])
            tile_gp = tl.dot(tile_x, tile_WT_gp.T, acc=tile_gp)
            # if has_bias_gp:
            #     tile_gp += ...

        ### compute act(tile_gp) * tile_up (skip dropout for now)
        tile_out = _act_fwd(tile_gp, act_fn) * tile_up

        out_desc.store([offset_m, offset_n], tile_out)


def mlp_hidden_states_fwd(
    x: Tensor,
    WT_up: Tensor,  # this one is transposed
    b_up: Tensor | None,
    WT_gp: Tensor,  # this one is transposed
    b_gp: Tensor | None,
    act_fn: str,
    dropout_p: float,
) -> Tensor:
    """
    This function computes the follwing operations in a fused fashion:

        self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

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

    ### Create the grid
    NUM_SMS = get_num_streaming_multiprocessors()
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 32, 2)
    # grid = (min(NUM_SMS, math.ceil(N / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),)
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    ### custom allocation function
    def allocator(size, stream: int, allignment: Optional[int]):
        return torch.empty(size, device=x.device, dtype=torch.int8)

    triton.set_allocator(allocator)

    ### allocate output and run the kernel
    out = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    has_b_gp = b_gp is not None
    has_b_up = b_up is not None
    with torch.cuda.device(x.device.index):
        _fwd_kernel[grid](
            x,
            WT_up,
            # b_up,
            # has_b_up,
            WT_gp,
            # b_gp,
            # has_b_gp,
            out,
            act_fn,
            dropout_p,
            M,
            N,
            K,
            NUM_SMS,
            # BLOCK_SIZE_M,
            # BLOCK_SIZE_N,
            # BLOCK_SIZE_K,
            # GROUP_SIZE_M,
        )

    ###
    return out.to(x.dtype) if x.dtype != out.dtype else out

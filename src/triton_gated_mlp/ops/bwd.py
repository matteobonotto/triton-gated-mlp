import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple, Optional
import math

import triton_dejavu

from .utils import get_num_streaming_multiprocessors, map_pid_m_n
from .act import _act_bwd, _act_fwd

def launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


# def get_autotune_configs(pre_hook=None):
#     return [
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS, 
#             }, pre_hook=pre_hook)  #
#         for BM in [16, 32, 64, 128]  #
#         for BN in [16, 32, 64, 128]  #
#         for BK in [16, 32, 64, 128]  #
#         for GS in [2, 4, 8]  #
#         # for s in ([2])  #
#         # for w in [4]  #
#         # for SUBTILE in [True, False]  #
#     ]

def get_autotune_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS, 
            }, pre_hook=pre_hook)  #
        for BM in [32, 64, 128, 256]  #
        for BN in [32, 64, 128, 256]  #
        for BK in [32, 64, 128, 256]  #
        for GS in [4, 8]  #
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
        {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS},
        # num_warps=[4, 8, 16],
        # num_stages=[1, 2, 4, 6],
    )

@triton_dejavu.autotune(
    # configs=get_autotune_configs(), 
    config_space=get_autotune_config_space(),
    key=["M", "N", "K"],
    use_bo=True,
)
@triton.jit(launch_metadata=launch_metadata)
def _compute_quantities_for_bwd(
    x_ptr,
    W_up_ptr,
    W_gp_ptr,
    grad_output_ptr,
    act_prime_ptr,
    grad_output_a_act_prime_ptr,
    grad_output_c_ptr,
    act_fun: tl.constexpr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing:

        a = x @ W_up.T
        b = x @ W_gp.T
        c = act_fwd(b)

        sigma = torch.sigmoid(b)
        act_prime = sigma * (1 + b * (1 - sigma))

        grad_output_a_act_prime = (grad_output * a) * act_prime
        grad_output_c = grad_output * c
    """
    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_total_programs = num_programs_m * num_programs_n

    ### create tensor desctiptors (inputs)
    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    W_up_desc = tl.make_tensor_descriptor(
        W_up_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    W_gp_desc = tl.make_tensor_descriptor(
        W_gp_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    grad_output_desc = tl.make_tensor_descriptor(
        grad_output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    ### create tensor desctiptors (outputs)
    act_prime_desc = tl.make_tensor_descriptor(
        act_prime_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_output_a_act_prime_desc = tl.make_tensor_descriptor(
        grad_output_a_act_prime_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_output_c_desc = tl.make_tensor_descriptor(
        grad_output_c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    ### loop along M
    for tile_id in tl.range(pid, num_total_programs, NUM_SMS):
        pid_m, pid_n = map_pid_m_n(
            tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, True
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_n = pid_n * BLOCK_SIZE_N

        tile_a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_b = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        ### loop along K
        for offset_k in tl.range(0, K, BLOCK_SIZE_K):
            # NOTE: masking along k is done under the hood by tensor descriptors
            # offset_k = k * BLOCK_SIZE_K

            tile_x = x_desc.load([offset_m, offset_k])
            tile_W_up = W_up_desc.load([offset_n, offset_k])
            tile_W_gp = W_gp_desc.load([offset_n, offset_k])

            tile_a = tl.dot(tile_x, tile_W_up.T, acc=tile_a)
            tile_b = tl.dot(tile_x, tile_W_gp.T, acc=tile_b)

        tile_c = _act_fwd(tile_b, act_fun)
        tile_act_prime = _act_bwd(tile_b, act_fun)

        # grad_output_a_act_prime = (grad_output * a) * act_prime
        tile_grad_output = grad_output_desc.load([offset_m, offset_n])
        tile_grad_output_a_act_prime = (tile_grad_output * tile_a) * tile_act_prime

        # grad_output_c = grad_output * c
        tile_grad_output_c = tile_grad_output * tile_c

        ### store results back in memory
        offsets = [offset_m, offset_n]
        act_prime_desc.store(offsets=offsets, value=tile_act_prime)
        grad_output_a_act_prime_desc.store(
            offsets=offsets, value=tile_grad_output_a_act_prime
        )
        grad_output_c_desc.store(offsets=offsets, value=tile_grad_output_c)



def get_autotune_config_space():
    BM = [32, 64, 128]  #
    BN = [32, 64, 128]  #
    BK = [32, 64, 128]  #
    GS = [4, 8, 16]
    return triton_dejavu.ConfigSpace(
        {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS},
        # num_warps=[4, 8, 16],
        # num_stages=[1, 2, 4, 6],
    )

@triton_dejavu.autotune(
    # configs=get_autotune_configs(), 
    config_space=get_autotune_config_space(),
    key=["M", "N", "K"],
    use_bo=True,
)

@triton_dejavu.autotune(
    configs=get_autotune_configs(), 
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=launch_metadata)
def _compute_dx(
    grad_output_c_ptr,
    W_up_ptr,
    grad_output_a_act_prime_ptr,
    W_gp_ptr,
    dx_ptr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel computing:
        
        dx_1 = grad_output_c @ W_up
        dx_2 = grad_output_a_act_prime @ W_gp
        dx = dx_1 + dx_2

    """
    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_programs = num_programs_m * num_programs_k

    ### tensor descriptors of inputs
    grad_output_c_desc = tl.make_tensor_descriptor(
        grad_output_c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    W_up_desc = tl.make_tensor_descriptor(
        W_up_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )
    grad_output_a_act_prime_desc = tl.make_tensor_descriptor(
        grad_output_a_act_prime_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )
    W_gp_desc = tl.make_tensor_descriptor(
        W_gp_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )

    ### tensor descriptor of output
    dx_desc = tl.make_tensor_descriptor(
        dx_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )

    for tile_id in tl.range(pid, num_programs, NUM_SMS):
        pid_m, pid_k = map_pid_m_n(
            tile_id, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K, GROUP_SIZE_M, True
        )

        offset_m = pid_m * BLOCK_SIZE_M
        offset_k = pid_k * BLOCK_SIZE_K

        tile_dx_1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        tile_dx_2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for offset_n in tl.range(0, N, BLOCK_SIZE_N):

            tile_grad_output_c = grad_output_c_desc.load([offset_m, offset_n])
            tile_W_up = W_up_desc.load([offset_n, offset_k])
            tile_dx_1 = tl.dot(tile_grad_output_c, tile_W_up, acc=tile_dx_1)

            tile_grad_output_a_act_prime = grad_output_a_act_prime_desc.load(
                [offset_m, offset_n]
            )
            tile_W_gp = W_gp_desc.load([offset_n, offset_k])
            tile_dx_2 = tl.dot(tile_grad_output_a_act_prime, tile_W_gp, acc=tile_dx_2)

        tile_dx = tile_dx_1 + tile_dx_2

        ### store back in memory
        dx_desc.store([offset_m, offset_k], tile_dx)


# def get_autotune_configs_dW(pre_hook=None):
#     return [
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS, 
#             }, pre_hook=pre_hook)  #
#         for BM in [16, 32]  #
#         for BN in [16, 32]  #
#         for BK in [32, 64, 128]  #
#         for GS in [2, 4, 8]  #
#         # for s in ([2])  #
#         # for w in [4]  #
#         # for SUBTILE in [True, False]  #
#     ]


def get_autotune_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS, 
            }, pre_hook=pre_hook)  #
        for BM in [16, 32, 64]  #
        for BN in [16, 32, 64]  #
        for BK in [32, 64, 128, 256]  #
        for GS in [4, 8]  #
        # for s in ([2])  #
        # for w in [4]  #
        # for SUBTILE in [True, False]  #
    ]

def get_autotune_config_space():
    BM = [8, 16, 32]  #
    BN = [16, 32, 64, 128]  #
    BK = [32, 64, 128, 256]  #
    GS = [2, 4, 8]
    return triton_dejavu.ConfigSpace(
        {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GS},
        # num_warps=[4, 8, 16],
        # num_stages=[1, 2, 4, 6],
    )


@triton_dejavu.autotune(
    # configs=get_autotune_configs(), 
    config_space=get_autotune_config_space(),
    key=["M", "N", "K"],
    use_bo=True,
)
@triton.jit(launch_metadata=launch_metadata)
def _compute_dW_up_dW_gp(
        x_ptr, grad_output_c_ptr, grad_output_a_act_prime_ptr, dW_up_ptr, dW_gp_ptr,
        M,
        N,
        K,
        NUM_SMS : tl.constexpr,
        BLOCK_SIZE_M : tl.constexpr,
        BLOCK_SIZE_K : tl.constexpr,
        BLOCK_SIZE_N : tl.constexpr,
        GROUP_SIZE_M : tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_programs = num_programs_n * num_programs_k

    ### make tensor descriptors inputs
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    grad_output_c_desc = tl.make_tensor_descriptor(
        grad_output_c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )
    grad_output_a_act_prime_desc = tl.make_tensor_descriptor(
        grad_output_a_act_prime_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    ### make tensor descriptors outputs
    dW_up_desc = tl.make_tensor_descriptor(
        dW_up_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )
    dW_gp_desc = tl.make_tensor_descriptor(
        dW_gp_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )

    ### persistent loop
    for tile_id in tl.range(pid, num_programs, NUM_SMS):
        pid_n, pid_k = map_pid_m_n(
            tile_id, N, K, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, True
        )

        offset_n = pid_n * BLOCK_SIZE_N
        offset_k = pid_k * BLOCK_SIZE_K

        # print(f"{offset_n=}, {offset_k=}")

        tile_dW_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), tl.float32)
        tile_dW_gp = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), tl.float32)

        for offset_m in tl.range(0, BLOCK_SIZE_M, M):

            tile_grad_output_c = grad_output_c_desc.load([offset_m, offset_n])
            tile_grad_output_a_act_prime = grad_output_a_act_prime_desc.load([offset_m, offset_n])
            tile_x = x_desc.load([offset_m, offset_k])

            tile_dW_up = tl.dot(tile_grad_output_c.T, tile_x, acc=tile_dW_up)
            tile_dW_gp = tl.dot(tile_grad_output_a_act_prime.T, tile_x, acc=tile_dW_gp)

        dW_up_desc.store(offsets=[offset_n, offset_k], value=tile_dW_up)
        dW_gp_desc.store(offsets=[offset_n, offset_k], value=tile_dW_gp)


def mlp_hidden_states_bwd(
    x: Tensor, W_up: Tensor, W_gp: Tensor, grad_output: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    act_fun = "silu"  # hard-coded for now, parametrize later

    M, K = x.shape
    N, _ = W_gp.shape
    # M = B * M # aready reshaped (B, M, K) -> (B * M, K)

    kwargs = {"device": x.device, "dtype": x.dtype}

    act_prime = torch.zeros((M, N), **kwargs)  # check dtype
    grad_output_a_act_prime = torch.zeros((M, N), **kwargs)  # check dtype
    grad_output_c = torch.zeros((M, N), **kwargs)  # check dtype

    NUM_SMS = get_num_streaming_multiprocessors()
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = (64, 64, 64, 8)

    ### provide the allocation function
    def allocation_fn(shape: int, stream: int, alligmnet: Optional[int]):
        return torch.empty(shape, device=x.device, dtype=torch.int8)

    triton.set_allocator(allocation_fn)

    # grid = (min(NUM_SMS, math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)),)
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    ### compute quantities for bwd
    _compute_quantities_for_bwd[grid](
        x,
        W_up,
        W_gp,
        grad_output,
        act_prime,
        grad_output_a_act_prime,
        grad_output_c,
        act_fun,
        M,
        N,
        K,
        NUM_SMS,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_N,
        # BLOCK_SIZE_K,
        # GROUP_SIZE_M,
    )

    """
    # steps to check the results
    a = x @ W_up.T
    b = x @ W_gp.T
    c = ACT_FWD[act_fn](b)
    sigma = torch.sigmoid(b)

    act_prime_ref = sigma * (1 + b * (1 - sigma))
    grad_output_a_act_prime_ref = (grad_output * a) * act_prime
    grad_output_c_ref = grad_output * c

    atol = 1e-3 if "TRITON_INTERPRET" in os.environ.keys() else 1e-7
    triton.testing.assert_close(act_prime, act_prime_ref, atol)
    triton.testing.assert_close(grad_output_a_act_prime, grad_output_a_act_prime_ref, atol)
    triton.testing.assert_close(grad_output_c, grad_output_c_ref, atol)
    """

    ### compute derivatives
    """
    ### dx
    dx_1 = grad_output_c @ W_up
    dx_2 = grad_output_a_act_prime @ W_gp
    dx = dx_1 + dx_2

    ### dW_up
    dW_up = grad_output_c.T @ x

    ### dW_gp
    dW_gp = grad_output_a_act_prime.T @ x
    
    """
    dx = torch.zeros(x.shape, **kwargs)  # .view() already done at higher level
    dW_up = torch.zeros(W_up.shape, **kwargs)  # check dtype
    dW_gp = torch.zeros(W_gp.shape, **kwargs)  # check dtype

    # grid = (min(NUM_SMS, math.ceil(M / BLOCK_SIZE_M) * math.ceil(K / BLOCK_SIZE_K)),)
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"])), )
    _compute_dx[grid](
        grad_output_c,
        W_up,
        grad_output_a_act_prime,
        W_gp,
        dx,
        M,
        N,
        K,
        NUM_SMS,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_K,
        # BLOCK_SIZE_N,
        # GROUP_SIZE_M,
    )

    grid = lambda META: (min(NUM_SMS, triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    # grid = (min(NUM_SMS, math.ceil(K / BLOCK_SIZE_K) * math.ceil(N / BLOCK_SIZE_N)),)
    _compute_dW_up_dW_gp[grid](
        x, grad_output_c, grad_output_a_act_prime, dW_up, dW_gp,
        M,
        N,
        K,
        NUM_SMS,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_K,
        # BLOCK_SIZE_N,
        # GROUP_SIZE_M,
    )

    return dx, dW_up, dW_gp

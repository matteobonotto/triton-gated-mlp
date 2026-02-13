import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Optional
import math


def get_target_dtype(x: Tensor) -> str:
    match x.dtype:
        case torch.float32:
            return "float32"
        case torch.bfloat16:
            return "bf16"
        case torch.float16:
            return "float16"
        case _:
            message = f"Got unexpected dtype: {x.dtype}"
            raise TypeError(message)


@triton.jit()
def map_dtype(x, dtype):
    match dtype:
        case "float32":
            return x.to(tl.float32)
        case "bf16":
            return x.to(tl.bfloat16)
        case "float16":
            return x.to(tl.float16)


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def cdiv(num: int, den: int) -> int:
    return math.ceil(num / den)


def get_shared_mem_limit(device: Optional[torch.device] = None) -> int:
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # This is the maximum dynamic shared memory per block (bytes)
    return props.shared_memory_per_block


def validate_dimensions_gmlp(
    hidden_states: Tensor,
    Wu: Tensor,  # this one is transposed
    Wg: Tensor,  # this one is transposed
    bu: Tensor | None = None,
    bg: Tensor | None = None,
    Wo: Tensor | None = None,  # this one is transposed
    bo: Tensor | None = None,
) -> None:
    assert (
        hidden_states.ndim <= 2
    ), f"input tensor must have ndims <=2, got {hidden_states.ndim}"
    assert Wu.shape[1] == hidden_states.shape[1], "dimension mismatch in Wu or x"
    assert Wu.shape == Wg.shape, "dimension mismatch in Wu or Wg"

    if Wo is not None:
        Wo.shape[0] == hidden_states.shape[1], "dimension mismatch in Wo or x"
        Wo.shape[1] == Wu.shape[0], "dimension mismatch in Wo or Wu"

    if bu is not None:
        assert bu.shape[0] == Wu.shape[0], "dimension mismatch in bu"

    if bg is not None:
        assert bg.shape[0] == Wg.shape[0], "dimension mismatch in bg"

    if bo is not None:
        assert bo.shape[0] == Wo.shape[0], "dimension mismatch in bo"


def pad_tensor_16_byte_aligned(t: Tensor, axis: int) -> Tensor:
    assert t.ndim == 2, f"expected tensor to have exactly 2 dimensions, got {t.ndims}"
    old_dims = t.shape
    dim = old_dims[axis]
    padded_dim = dim + 16 - dim % 16
    new_dims = (padded_dim, t.shape[1]) if axis == 0 else (t.shape[0], padded_dim)
    new_t = torch.zeros(new_dims, dtype=t.dtype, device=t.device)
    new_t[: old_dims[0], : old_dims[1]] = t
    return new_t


def get_num_streaming_multiprocessors() -> int:
    return (
        2  # dummy value for dev/debugging
        if not torch.cuda.is_available()
        else torch.cuda.get_device_properties("cuda:0").multi_processor_count
    )


@triton.jit()
def map_pid_m_n(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, optimize_L2):
    if optimize_L2:
        pid_m, pid_n = map_pid_m_n_L2_optim(
            pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )
    else:
        n_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // n_programs_n
        pid_n = pid % n_programs_n
    return (pid_m, pid_n)


@triton.jit()
def map_pid_m_n_L2_optim(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):

    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_in_group = GROUP_SIZE_M * num_blocks_n

    group_size_m = GROUP_SIZE_M
    offset_m = pid // num_programs_in_group
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - offset_m * GROUP_SIZE_M)

    pid_m = ((pid % num_programs_in_group) % group_size_m) + offset_m * GROUP_SIZE_M
    pid_n = (pid % num_programs_in_group) // group_size_m

    return (pid_m, pid_n)

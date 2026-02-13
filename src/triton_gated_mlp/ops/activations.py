import triton
import triton.language as tl


@triton.jit()
def _act_fwd(x, act_name: tl.constexpr):
    if act_name == "no_act":
        return x
    elif act_name == "silu":
        return _silu_fwd(x)
    else:
        raise NotImplementedError()


@triton.jit()
def _act_bwd(x, act_name: tl.constexpr):
    if act_name == "no_act":
        return x
    elif act_name == "silu":
        return _silu_bwd(x)
    else:
        raise NotImplementedError()


# -------------------- activations / fwd --------------------


@triton.jit()
def _compute_sigma(x):
    return 1 / (1 + tl.exp(-x))


@triton.jit()
def _silu_fwd(x):
    return x * _compute_sigma(x)


# -------------------- activations / bwd --------------------


@triton.jit()
def _silu_bwd(x):
    sigma = _compute_sigma(x)
    act_prime = sigma * (1 + x * (1 - sigma))
    return act_prime

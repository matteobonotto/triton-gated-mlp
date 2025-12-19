from triton_kernels.nn.gated_mlp.gated_mlp import (
    NaiveGatedMLP,
    FusedGatedMLP,
    eager_fwd,
    mlp_hidden_states_fwd,
    mlp_hidden_states_bwd,
    ACT_FWD,
)
from triton_kernels.utils import get_device

from torch.autograd import grad
import triton
from torch import nn, Tensor
import torch
import os


def copy_weights(module_src: nn.Module, module_tgt: nn.Module) -> None:

    for (n1, p1), (n2, p2) in zip(
        module_src.named_parameters(), module_tgt.named_parameters()
    ):
        assert n1 == n2, "name mismatch"
        p2.data = p1.data.clone()


def test_gated_mlp_bwd():

    DEVICE = get_device()
    DTYPE = torch.float32

    gmlp_1 = NaiveGatedMLP(dropout_p=0.0, bias=False).to(DEVICE).to(DTYPE)
    gmlp_2 = FusedGatedMLP(dropout_p=0.0, bias=False).to(DEVICE).to(DTYPE)

    copy_weights(gmlp_1, gmlp_2)

    M = 32
    K = gmlp_1.hidden_size
    B = 16

    x = torch.rand((B, M, K), device=DEVICE, dtype=DTYPE)
    x.requires_grad = True

    ### test fwd pass first
    out_1 = gmlp_1(x)
    out_2 = gmlp_2(x)
    print((out_1 - out_2).norm() / out_1.norm())
    triton.testing.assert_close(
        out_1, out_2, atol=1e-6 if "TRITON_INTERPRET" in os.environ.keys() else 1e-3
    )

    ### then test bwd pass
    grad_outputs = torch.rand_like(x).to(DEVICE)
    inputs = (
        x,
        gmlp_1.up_proj.weight,
        # gmlp_1.up_proj.bias,
        gmlp_1.gate_proj.weight,
        # gmlp_1.gate_proj.bias,
        # gmlp_1.act_fn,
        # gmlp_1.dropout.p,
    )
    grads_1 = grad(out_1, inputs, grad_outputs=grad_outputs, retain_graph=True)
    inputs = (
        x,
        gmlp_2.up_proj.weight,
        # gmlp_2.up_proj.bias,
        gmlp_2.gate_proj.weight,
        # gmlp_2.gate_proj.bias,
        # gmlp_2.act_fn,
        # gmlp_2.dropout.p,
    )
    grads_2 = grad(
        out_2, inputs, grad_outputs=grad_outputs, retain_graph=True, allow_unused=True
    )
    for g1, g2 in zip(grads_1, grads_2):
        print((g1 - g2).norm() / g2.norm())
    print("Done!!")

    triton.testing.assert_close(gmlp_1(x), gmlp_2(x), rtol=1e-6)


test_gated_mlp_bwd()


def assert_close(x: Tensor, x_ref: Tensor, atol: float):
    norm_diff = ((x - x_ref).norm() / x_ref.norm()).item()
    assert norm_diff < atol, f"Got 'norm_diff' >= 'atol, where {norm_diff=}, {atol=}"
    print(f"{norm_diff=:3.3}, {atol=}")


def test_bwd_ops_triton():

    DEVICE = get_device()
    DTYPE = torch.float32
    kwargs = {"device": DEVICE, "dtype": DTYPE}
    atol = 1e-3 if "TRITON_INTERPRET" in os.environ.keys() else 1e-7

    B, M, N, K = (2, 32, 512, 256)
    BM = B * M

    x = torch.rand((BM, K), **kwargs) / K
    W_up = torch.rand((N, K), **kwargs) / K
    W_gp = torch.rand((N, K), **kwargs) / K
    hidden_states = torch.rand((BM, N), **kwargs) / K
    grad_output = torch.rand(hidden_states.shape, **kwargs) / K

    ### ref quantities
    act_fn = "silu"

    a = x @ W_up.T
    b = x @ W_gp.T
    c = ACT_FWD[act_fn](b)
    sigma = torch.sigmoid(b)

    act_prime = sigma * (1 + b * (1 - sigma))
    grad_output_a_act_prime = (grad_output * a) * act_prime
    grad_output_c = grad_output * c

    dx_1 = grad_output_c @ W_up
    dx_2 = grad_output_a_act_prime @ W_gp
    dx_ref = dx_1 + dx_2

    dW_up_ref = grad_output_c.T @ x

    dW_gp_ref = grad_output_a_act_prime.T @ x

    ### triton kernel
    dx, dW_up, dW_gp = mlp_hidden_states_bwd(x, W_up, W_gp, grad_output)
    assert_close(dx, dx_ref, atol)
    assert_close(dW_up, dW_up_ref, atol)
    assert_close(dW_gp, dW_gp_ref, atol)



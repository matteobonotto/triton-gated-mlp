from utils import create_tensors
import random
import torch
from torch import autograd

from triton_gated_mlp.utils import get_device

DEVICE = get_device()


def norm_diff(a, b):
    return (a - b).norm() / a.norm()


def test_eager_bwd():

    (
        x,
        Wu,
        bu,
        Wg,
        bg,
    ) = create_tensors(
        M=random.randint(16, 256),
        N=random.randint(16, 256),
        K=random.randint(16, 256),
        DEVICE=DEVICE,
        requires_grad=True,
    )

    p = 0.0

    act = torch.nn.functional.silu
    dropout = torch.nn.functional.dropout

    xu = x @ Wu.T + bu
    xg = x @ Wg.T + bg
    xa = act(xg)
    xp = xu * xa
    y = dropout(xp, p=p)

    grad_outputs = torch.rand_like(xp)

    dx, dWu, dbu, dWg, dbg = autograd.grad(
        xp, (x, Wu, bu, Wg, bg), grad_outputs=grad_outputs
    )

    dropout_prime = torch.ones_like(grad_outputs, dtype=x.dtype, device=x.device)

    # dWu, dbu
    dWu_ = ((grad_outputs * dropout_prime) * xa).T @ x
    dbu_ = ((grad_outputs * dropout_prime) * xa).sum(dim=-2)

    # dWg, dbg
    sigma = 1 / (1 + torch.exp(-xg))
    act_prime = sigma * (1 + xg * (1 - sigma))
    dWg_ = (((grad_outputs * dropout_prime) * xu) * act_prime).T @ x
    dbg_ = (((grad_outputs * dropout_prime) * xu) * act_prime).sum(dim=-2)

    # dx
    dx_tilde = grad_outputs * dropout_prime
    dx_ = (dx_tilde * xu * act_prime) @ Wg + (dx_tilde * xa) @ Wu

    assert norm_diff(dWu, dWu_) < 1e-6
    assert norm_diff(dbu, dbu_) < 1e-6
    assert norm_diff(dWg, dWg_) < 1e-6
    assert norm_diff(dbg, dbg_) < 1e-6
    assert norm_diff(dx, dx_) < 1e-6


def test_bwd():

    (
        x,
        Wu,
        bu,
        Wg,
        bg,
    ) = create_tensors(
        M=random.randint(16, 256),
        N=random.randint(16, 256),
        K=random.randint(16, 256),
        DEVICE=DEVICE,
        requires_grad=True,
    )

    p = 0.0

    act = torch.nn.functional.silu
    dropout = torch.nn.functional.dropout

    xu = x @ Wu.T + bu
    xg = x @ Wg.T + bg
    xa = act(xg)
    xp = xu * xa
    y = dropout(xp, p=p)

    grad_outputs = torch.rand_like(xp)

    dx, dWu, dbu, dWg, dbg = autograd.grad(
        xp, (x, Wu, bu, Wg, bg), grad_outputs=grad_outputs
    )

    assert norm_diff(dWu, dWu_) < 1e-6
    assert norm_diff(dbu, dbu_) < 1e-6
    assert norm_diff(dWg, dWg_) < 1e-6
    assert norm_diff(dbg, dbg_) < 1e-6
    assert norm_diff(dx, dx_) < 1e-6


test_bwd()

from utils import create_tensors
import random
import torch
from torch import autograd

DEVICE = torch.device("cuda")


def norm_diff(a, b):
    return (a - b).norm() / a.norm()


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

    assert norm_diff(dWu, (grad_outputs * xa).T @ x) < 1e-6


test_bwd()

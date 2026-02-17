from triton_gated_mlp.gated_mlp import (
    EagerGatedMLP,
    FusedGatedMLP,
    mlp_hidden_states_fwd,
    eager_fwd,
)
from triton_gated_mlp.utils import get_device

import triton
from torch import nn
import torch


def copy_weights(module_src: nn.Module, module_tgt: nn.Module) -> None:

    for (n1, p1), (n2, p2) in zip(
        module_src.named_parameters(), module_tgt.named_parameters()
    ):
        assert n1 == n2, "name mismatch"
        p2.data = p1.data.clone()


def test_gated_mlp_fwd():

    DEVICE = get_device()
    DTYPE = torch.float32

    gmlp_1 = EagerGatedMLP(dropout_p=0.0).to(DEVICE).to(DTYPE)
    gmlp_2 = FusedGatedMLP(dropout_p=0.0).to(DEVICE).to(DTYPE)

    copy_weights(gmlp_1, gmlp_2)

    M = 128
    K = gmlp_1.hidden_size

    x = torch.rand((M, K), device=DEVICE, dtype=DTYPE)

    triton.testing.assert_close(gmlp_1(x), gmlp_2(x), rtol=1e-6)


def test_fwd_op_triton():

    DEVICE = get_device()
    DTYPE = torch.float32

    init_args = {
        "hidden_act": "no_act",
        "dropout_p": 0.0,
        "bias": False,
    }

    gmlp = EagerGatedMLP(**init_args).to(DEVICE).to(DTYPE)

    M = 128
    K = gmlp.hidden_size

    x = torch.rand((M, K), device=DEVICE, dtype=DTYPE)
    out_ref = eager_fwd(
        x,
        gmlp.up_proj.weight,
        gmlp.up_proj.bias,
        gmlp.gate_proj.weight,
        gmlp.gate_proj.bias,
        # gmlp.act_fn,
        # gmlp.dropout.p,
    )
    print(out_ref)

    out_triton = mlp_hidden_states_fwd(
        x,
        gmlp.up_proj.weight,
        gmlp.up_proj.bias,
        gmlp.gate_proj.weight,
        gmlp.gate_proj.bias,
        gmlp.act_fn,
        gmlp.dropout.p,
    )
    print(out_triton)

    print((out_ref - out_triton).norm() / out_ref.norm())

    triton.testing.assert_close(out_ref, out_triton, rtol=1e-6)

    ...

    # triton.testing.assert_close(gmlp_1(x), gmlp_2(x), rtol=1e-6)


test_fwd_op_triton()

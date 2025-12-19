from typing import Dict, Optional
from torch import nn
import torch
from collections import OrderedDict

from .ops.fwd import mlp_hidden_states_fwd
from .ops.bwd import mlp_hidden_states_bwd


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

ACT_FWD = {
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "no_act": nn.Identity(),
}


def _silu_op_bwd(x, grad_output):
    sigma = 1 / (1 + torch.exp(-x))
    act_prime = sigma * (1 + x * (1 - sigma))
    return grad_output * act_prime


ACT_BWD = {
    # "gelu": nn.GELU(),
    # "leaky_relu": nn.LeakyReLU(),
    # "relu": nn.ReLU(),
    # "sigmoid": nn.Sigmoid(),
    "silu": _silu_op_bwd,
    # "swish": nn.SiLU(),
    # "tanh": nn.Tanh(),
    # "no_act": nn.Identity(),
}


# def mlp_fwd(x, W1, W2, W3):
#     return (torch.nn.functional.silu(x @ W2.T) * (x @ W1.T)) @ W3.T

# class LlamaMLP(nn.Module):
#     def __init__(self, hidden_size: int = 64, intermediate_size: int = 256):
#         super().__init__()
#         self.W1 = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.W2 = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.W3 = nn.Linear(intermediate_size, hidden_size, bias=False)
#         self.act = nn.SiLU()

#     def forward(self, x: Tensor) -> Tensor:
#         return mlp_fwd(x, self.W1.weight, self.W2.weight, self.W3.weight)


class NaiveGatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        bias: bool = True,
        hidden_act: str = "silu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act = ACT_FWD[hidden_act]
        self.act_fn = hidden_act
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        shapes = x.shape
        if len(shapes) > 2:
            x = x.view(-1, shapes[-1])
        hidden_states = self.dropout(self.act(self.gate_proj(x)) * self.up_proj(x))
        down_proj = self.down_proj(hidden_states)
        return down_proj.view(shapes) if len(shapes) > 2 else down_proj


from torch import Tensor
from torch.autograd.function import Function


def eager_fwd(
    x: Tensor,
    WT_up: Tensor,
    b_up: Optional[Tensor],
    WT_gp: Tensor,
    b_gp: Optional[Tensor],
    # act_fn: str,
    # dropout_p: float,
) -> Tensor:

    act_fn = "silu"

    up = x @ WT_up.T
    if b_up is not None:
        up += b_up

    gated = x @ WT_gp.T
    if b_gp is not None:
        gated += b_gp

    gated = ACT_FWD[act_fn](gated)
    out = gated * up
    # x = nn.functional.dropout(x, dropout_p, training)
    return out


def eager_bwd(x, W_up, W_gp, grad_output):
    act_fn = "silu"

    ### compute fwd pass quantities
    a = x @ W_up.T
    b = x @ W_gp.T
    c = ACT_FWD[act_fn](b)
    sigma = torch.sigmoid(b)

    ### some quantities for bwd computation
    act_prime = sigma * (1 + b * (1 - sigma))
    grad_output_a_act_prime = (grad_output * a) * act_prime
    grad_output_c = grad_output * c

    ### dx
    dx_1 = grad_output_c @ W_up
    dx_2 = grad_output_a_act_prime @ W_gp
    dx = dx_1 + dx_2

    ### dW_up
    dW_up = grad_output_c.T @ x

    ### dW_gp
    dW_gp = grad_output_a_act_prime.T @ x

    return dx, dW_up, dW_gp


class FusedGatedMLPFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        W_up: Tensor,
        b_up: Tensor,
        W_gp: Tensor,
        b_gp: Tensor,
        # act_fn: str,
        # dropout_p: float,
    ) -> Tensor:
        # hidden_states = eager_fwd(x, W_up, b_up, W_gp, b_gp)
        hidden_states = mlp_hidden_states_fwd(
            x=x,
            WT_up=W_up,
            b_up=None,
            WT_gp=W_gp,
            b_gp=None,
            act_fn="silu",
            dropout_p=0.0,
        )

        ctx.save_for_backward(x, W_up, W_gp)

        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        x, W_up, W_gp = ctx.saved_tensors

        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # print(f"{x.shape=}")
        # print(f"{W_up.shape=}")
        # print(f"{W_gp.shape=}")
        # print(f"{grad_output.shape=}")

        # dx, dW_up, dW_gp = eager_bwd(x, W_up, W_gp, grad_output)
        dx, dW_up, dW_gp = mlp_hidden_states_bwd(x, W_up, W_gp, grad_output)

        return dx, dW_up, None, dW_gp, None


class FusedGatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        bias: bool = True,
        hidden_act: str = "silu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act = ACT_FWD[hidden_act]
        self.act_fn = hidden_act
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        # hidden_states = self.dropout(self.act(self.gate_proj(x)) * self.up_proj(x))
        shapes = x.shape
        if len(shapes) > 2:
            x = x.view(-1, shapes[-1])
        hidden_states = FusedGatedMLPFunction.apply(
            x,
            self.up_proj.weight,
            self.up_proj.bias,
            self.gate_proj.weight,
            self.gate_proj.bias,
            # self.act_fn,
            # self.dropout.p,
        )
        down_proj = self.down_proj(hidden_states)
        return down_proj.view(shapes) if len(shapes) > 2 else down_proj

import triton
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor

from triton_gated_mlp.gated_mlp import FusedGatedMLP, EagerGatedMLP
from triton_gated_mlp.utils import get_device


if __name__ == "__main__":
    providers = ["torch", "triton"]

    available_colors = [
        "black",
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "purple",
        "grey",
    ]
    colors = available_colors[: len(providers)]

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["N", "M", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[
                int(2**i) for i in np.arange(5, 8, 1)
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=providers,  # ["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=providers,  # ["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[
                (x, "-") for x, _ in zip(colors, providers)
            ],  # ("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="gated-mlp-performance",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )

    def step(mlp, x):
        out = mlp(x)
        loss = out.sum()
        loss.backward()

    MAP_MODULE = {
        "torch": EagerGatedMLP,
        "triton": FusedGatedMLP,
    }

    DTYPE = torch.float16
    DEVICE = get_device()

    def get_mlp(provider, N, K, DEVICE, DTYPE):
        gmlp_kwargs = {
            "hidden_size": K,
            "intermediate_size": N,
            "dropout_p": 0.0,
            "bias": False,
        }
        return MAP_MODULE[provider](**gmlp_kwargs).to(DEVICE).to(DTYPE)

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):

        M *= 4
        K *= 2

        gmlp = get_mlp(provider, N, K, DEVICE, DTYPE)

        print(f"{provider=}, {DTYPE=}, {M=}, {N=}, {K=}")

        x = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        x.requires_grad = True

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: step(gmlp, x), quantiles=quantiles
        )
        perf = lambda ms: 9 * 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    df_ops: pd.DataFrame = benchmark.run(
        show_plots=True, print_data=True, return_df=True
    )[0]
    ax = df_ops.plot(x="N", kind="bar", y=["torch", "triton"], ylabel="TFLOPS")
    ax.figure.tight_layout()
    ax.figure.savefig("benchmark_tflops.png")

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):

        M *= 4
        K *= 2

        gmlp = get_mlp(provider, N, K, DEVICE, DTYPE)

        # Initialize input
        x = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
        x.requires_grad = True
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()  # Optional: clear fragmentation
        step(gmlp, x)
        max_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        max_memory_mb = max_memory_bytes / 1024 / 1024

        # Triton bench expects (y, y_min, y_max).
        # Since memory is deterministic (mostly), min/max are the same.
        return max_memory_mb, max_memory_mb, max_memory_mb

    df_mem: pd.DataFrame = benchmark.run(
        show_plots=True, print_data=True, return_df=True
    )[0]
    ax = df_mem.plot(x="N", kind="bar", y=["torch", "triton"], ylabel="TFLOPS")
    ax.figure.tight_layout()
    ax.figure.savefig("benchmark_memory.png")

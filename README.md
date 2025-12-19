# triton-gated-mlp

Triton implementation of the gated-mlp module. We use `triton-dejavu` for better caching of the autotuning configs, so you don't need to run the autotune each time.


## Installation
Just do the following steps:
1. clone the repo and `cd triton-gated-mlp`
1. `make install` to set-up all the stuff and create the vistualenv
2. `make setup-triton-dejavu` this step is required to properly set-up `triton-dejavu`, providing a folder with proper permissions to cache the optimal parameters found by the autotuner.
3. `make install` to install all the project dependencies and the package

## Autotuning
It is strongly suggested to run, as first thing after the installation, the command
```shell
make autotune
```
this will run the autotuner over a large parameter space for the matrix-matrix multiplications, finding the optimal parameters for different matrix dimensions.

## Usage
A simple example:
```python

import torch
from triton_gated_mlp.gated_mlp import FusedGatedMLP

M = 128 # sequence lenght
K = 512 # hidden size
N = 256 # intermediate size

x = torch.rand((M, K), device="cuda", dtype=torch.float16)
gnlp = FusedGatedMLP(
    hidden_size=K, 
    intermediate_size=N,
    dropout_p=0.0, 
    bias=False,
)

out = gmlp(x)

```

## Benchmarks
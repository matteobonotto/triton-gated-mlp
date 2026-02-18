import torch


def nice_tensor(n1, n2):
    return torch.arange(1, n1 * n2 + 1, dtype=torch.float32).reshape(n1, n2)


def create_tensors(M, N, K, DEVICE, requires_grad=False):
    x = torch.rand(M, K, requires_grad=requires_grad).to(DEVICE)
    Wu = torch.rand(N, K, requires_grad=requires_grad).to(DEVICE)
    bu = torch.rand(N, requires_grad=requires_grad).to(DEVICE)
    Wg = torch.rand(N, K, requires_grad=requires_grad).to(DEVICE)
    bg = torch.rand(N, requires_grad=requires_grad).to(DEVICE)

    return (
        x,
        Wu,
        bu,
        Wg,
        bg,
    )

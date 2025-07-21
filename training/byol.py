import torch


def compute_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean-squared error

    Args:
        a: input tensor 1
        b: input tensor 2

    Returns:
        Mean-squared error between a and b.
    """
    return ((a - b) ** 2).sum(dim=-1).mean()

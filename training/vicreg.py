import torch


def variance_loss(x_1: torch.Tensor, x_2: torch.Tensor, mach_eps: float) -> torch.Tensor:
    # TODO docs
    """

    Args:
        x_1:
        x_2:
        mach_eps:

    Returns:

    """
    x_1 = x_1 - x_1.mean(dim=0)
    x_2 = x_2 - x_2.mean(dim=0)
    std_x_1 = torch.sqrt(x_1.var(dim=0) + mach_eps)
    std_x_2 = torch.sqrt(x_2.var(dim=0) + mach_eps)
    return torch.mean(torch.nn.functional.relu(1 - std_x_1)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_x_2)) / 2


def covariance_loss(x_1: torch.Tensor, x_2: torch.Tensor, num_features: int, len_x: int) -> torch.Tensor:
    # TODO docs
    """

    Args:
        x_1:
        x_2:
        num_features:
        len_x:

    Returns:

    """
    cov_x_1 = (x_1.T @ x_1) / (len_x - 1)
    cov_x_2 = (x_2.T @ x_2) / (len_x - 1)
    return off_diagonal(cov_x_1).pow_(2).sum().div(num_features) + off_diagonal(cov_x_2).pow_(2).sum().div(num_features)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # TODO docs
    """

    Args:
        x:

    Returns:

    """
    # Source:
    # https://github.com/facebookresearch/vicreg/blob/a73f567660ae507b0667c68f685945ae6e2f62c3/main_vicreg.py#L239
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
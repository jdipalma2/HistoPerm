from typing import Tuple

import torch


def info_nce_loss(features: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the InfoNCE loss on the given features.

    Args:
        features: Features to compute loss on
        batch_size: Mini-batch size of the features.

    Returns:
        InfoNCE loss on the given features.
    """
    labels = torch.cat([torch.arange(batch_size) for __ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = torch.nn.functional.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.int64).to(features.device)

    return logits, labels

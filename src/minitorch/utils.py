from minitorch.tensor import Tensor
import numpy as np


def cross_entropy_loss(logits: Tensor, labels: list[int]) -> Tensor:
    assert len(logits.shape) == 2
    bs, n_classes = logits.shape

    one_hot_encodings = Tensor(np.eye(n_classes)[labels])

    scaling = logits.exp().sum(dim=1)
    return ((logits - scaling.log()) * one_hot_encodings).sum(dim=1).mean() * Tensor(
        np.ones((1, 1)) * -1
    )

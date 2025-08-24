from typing import Protocol

import torch
from torch.nn.functional import cross_entropy


class Metric(Protocol):
    def __call__(self, predictions: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        ...


def accuracy(scores, labels):
    return torch.eq(scores.argmax(dim=1), labels).float()


def cross_entropy_with_temperature(scores, labels, *, temperature: float):
    return cross_entropy(scores / temperature, labels) * temperature



from typing import Protocol, Annotated

import torch
from torch.nn.functional import cross_entropy



class _MetaAbstractDtype(type):
    def __getitem__(cls, item):
        # return cls
        return torch.Tensor


class Tensor(metaclass=_MetaAbstractDtype):
    pass


class FloatTensor(metaclass=_MetaAbstractDtype):
    pass


class IntegerTensor(metaclass=_MetaAbstractDtype):
    pass



class Metric(Protocol):
    def __call__(self,
                 prediction_scores: FloatTensor["shape [b, c]"],
                 s: FloatTensor,
                 labels: IntegerTensor["shape [b]"]
                 ) -> FloatTensor["shape [b]"]:
        ...


def accuracy(scores, labels):
    return torch.eq(scores.argmax(dim=1), labels).float()


def cross_entropy_with_temperature(scores, labels, *, temperature: float):
    return cross_entropy(scores / temperature, labels) * temperature



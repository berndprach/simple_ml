import random
from collections.abc import Sequence
from typing import Iterable, Callable

import torch


def shuffle(i: Iterable) -> Iterable:
    """ Only use on reasonable small iterables (or lists). """
    if not isinstance(i, Sequence):
        i = list(i)
    try:
        random.shuffle(i)
    except TypeError:  # Immutable
        i = list(i)
        random.shuffle(i)
    yield from i


def repeat(i: Iterable, n: int):
    if not isinstance(i, Sequence):
        i = list(i)
    for _ in range(n):
        yield from i


def batch(i: Iterable, batch_size) -> Iterable:
    current_batch = []
    for element in i:
        current_batch.append(element)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if len(current_batch) > 0:
        yield current_batch


def batch_tensor(i: Iterable, batch_size) -> Iterable:
    for element_batch in batch(i, batch_size):
        if isinstance(element_batch, torch.Tensor):
            yield element_batch
        # [(x1, y1), (x2, y3), ...] -> [stack(x1, x2, ..), stack(y1, y2, ...)]
        yield (torch.stack(x_list) for x_list in zip(*element_batch))


def to_device(i: Iterable, device) -> Iterable:
    # E.g. device = "cuda" if torch.cuda.is_available() else "cpu"
    for element_tuple in i:
        if isinstance(element_tuple, torch.Tensor):
            yield element_tuple.to(device)
        else:
            yield (element.to(device) for element in element_tuple)


def convert_x_to(i: Iterable, dtype) -> Iterable:
    # E.g. dtype = torch.float32
    for xy in i:
        x, *rest = xy
        yield x.to(dtype), *rest


def rescale_x(i: Iterable, factor) -> Iterable:
    for xy in i:
        x, *rest = xy
        yield x * factor, *rest

def center_x(i: Iterable, mean) -> Iterable:
    # Mean: torch.tensor with matching dimensions for broadcasting!
    for xy in i:
        x, *rest = xy
        yield x - mean, *rest


def apply_to_x(i: Iterable, fn: Callable) -> Iterable:
    for (x, *rest) in i:
        yield fn(x), *rest


class DataIterator:
    def __init__(self, data: list):
        self.data = data
        self.iterator = iter(data)

    def __iter__(self):
        yield from self.iterator

    def shuffle(self):
        self.iterator = shuffle(self.iterator)
        return self

    def repeat(self, n):
        self.iterator = repeat(self.iterator, n)
        return self

    def batch(self, batch_size):
        self.iterator = batch_tensor(self.iterator, batch_size)
        return self

    def batch_tensor(self, batch_size):
        self.iterator = batch_tensor(self.iterator, batch_size)
        return self

    def to(self, device):
        self.iterator = to_device(self.iterator, device)
        return self

    def convert_x_to(self, dtype):
        self.iterator = convert_x_to(self.iterator, dtype)
        return self

    def rescale_x(self, factor):
        self.iterator = rescale_x(self.iterator, factor)
        return self

    def center_x(self, mean):
        self.iterator = center_x(self.iterator, mean)
        return self

    def apply_to_x(self, fn):
        self.iterator = apply_to_x(self.iterator, fn)
        return self
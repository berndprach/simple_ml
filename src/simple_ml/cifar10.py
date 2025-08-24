import os
from functools import partial
from itertools import pairwise
from typing import Iterable

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor

ROOT = "../data"
CIFAR10 = partial(torchvision.datasets.CIFAR10, root="../data", transform=pil_to_tensor, target_transform=torch.tensor)

channel_means = (0.4914, 0.4822, 0.4465)
means = torch.tensor(channel_means)[:, None, None]


def load_data(use_test_data=False):
    if use_test_data:
        return load_train_test_data()
    else:
        return load_train_val_data()


def load_train_test_data():
    return (
        load_torchvision_data(train=True),
        load_torchvision_data(train=False),
    )


def load_train_val_data():
    train_val_data = load_torchvision_data(train=True)
    return split_deterministically(train_val_data, [45_000, 5_000])



def load_torchvision_data(train):
    try:
        return CIFAR10(train=train)
    except RuntimeError:
        print("CIFAR10 not found, downloading...")
        return CIFAR10(train=train, download=True)


def split_deterministically(data: list, split_sizes: Iterable[int], shuffle=True) -> list[list]:
    assert sum(split_sizes) <= len(data), "Split sizes exceed dataset size."
    indices = [i for i in range(len(data))]

    np.random.seed(1111)
    if shuffle:
        np.random.shuffle(indices)

    split_locations = [0]
    for s in split_sizes:
        split_locations.append(split_locations[-1] + s)

    split_data = []
    for s, t in pairwise(split_locations):
        split_data.append([data[i] for i in indices[s:t]])
    return split_data

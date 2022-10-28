import pytest

import torch

from src.metrics.counting_mae_metric import CountingMAEMetric
from src.metrics.dice_metric import DiceMetric


def test_PeopleMAE():
    metric = CountingMAEMetric()
    zeros = torch.zeros((4, 1, 32, 32))
    dest = torch.zeros((4, 1, 32, 32))
    dest[:, 0, 1, :5] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(5.0))
    del metric

    metric = CountingMAEMetric()
    zeros = torch.zeros((4, 1, 32, 32))
    dest = torch.zeros((4, 1, 32, 32))
    dest[0, 0, 1, :2] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(0.5))

    dest[1, 0, 1, 10:] = 100
    metric.update(dest, zeros)
    assert torch.isclose(metric.compute(), torch.tensor(3.25))


def test_DiceMetric():
    metric = DiceMetric()

    zeros = torch.zeros((4, 1, 32, 32))
    zeros[:, 0, 1, :5] = 100
    dest = torch.zeros((4, 1, 32, 32))
    dest[:, 0, 1, :5] = 100
    metric.update(dest, zeros)
    assert torch.isclose(torch.tensor(metric.compute()), torch.tensor(100.0))
    del metric

    metric = DiceMetric()
    zeros = torch.zeros((4, 1, 32, 32))
    zeros[:, 0, 1, :1] = 100
    dest = torch.zeros((4, 1, 32, 32))
    dest[:, 0, 1, :0] = 100
    metric.update(dest, zeros)
    assert torch.isclose(torch.tensor(metric.compute()), torch.tensor(0.0))

    zeros = torch.zeros((4, 1, 32, 32))
    zeros[:, 0, 1, :4] = 100
    dest = torch.zeros((4, 1, 32, 32))
    dest[:, 0, 1, :6] = 100
    metric.update(dest, zeros)
    assert torch.isclose(torch.tensor(metric.compute()), torch.tensor(40.0))


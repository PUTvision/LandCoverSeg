import torch

from src.losses.counting_mae_loss import CountingMAELoss
from src.losses.dice_loss import DiceLoss
from src.losses.relative_counting_mae_loss import RelativeCountingMAELoss


def test_CountingMAELoss():
    loss = CountingMAELoss()

    t0 = torch.zeros(1, 1, 200, 200)
    assert torch.isclose(loss.forward(t0, t0), torch.tensor(0.))

    t1 = torch.zeros_like(t0)
    t1[0, 0, 100, 100] = 100
    assert torch.isclose(loss.forward(t0, t1), torch.tensor(1.))
    assert torch.isclose(loss.forward(t1, t0), torch.tensor(1.))

    t2 = torch.zeros_like(t0)
    t2[0, 0, 100, 100] = 100
    t2[0, 0, 50, 100] = 100
    t2[0, 0, 100, 50] = 100

    assert torch.isclose(loss.forward(t0, t2), torch.tensor(3.))
    assert torch.isclose(loss.forward(t2, t0), torch.tensor(3.))

    t3 = torch.zeros_like(t0)
    t3[0, 0, 100, 100] = 100
    t3[0, 0, 50, 100] = 100

    t4 = torch.zeros_like(t0)
    t4[0, 0, 100, 100] = 100
    t4[0, 0, 100, 50] = 100

    assert torch.isclose(loss.forward(t3, t4), torch.tensor(0.))
    assert torch.isclose(loss.forward(t4, t3), torch.tensor(0.))


def test_RelativeCountingMAELoss():
    loss = RelativeCountingMAELoss()

    t0 = torch.zeros(1, 1, 200, 200)
    assert torch.isclose(loss.forward(t0, t0), torch.tensor(0.))

    t1 = torch.zeros_like(t0)
    t1[0, 0, 100, 100] = 100
    assert torch.isclose(loss.forward(t0, t1), torch.tensor(0.5))
    assert torch.isclose(loss.forward(t1, t0), torch.tensor(1.0))

    t2 = torch.zeros_like(t0)
    t2[0, 0, 100, 100] = 100
    t2[0, 0, 50, 100] = 100
    t2[0, 0, 100, 50] = 100

    assert torch.isclose(loss.forward(t0, t2), torch.tensor(0.75))
    assert torch.isclose(loss.forward(t2, t0), torch.tensor(3.))

    t3 = torch.zeros_like(t0)
    t3[0, 0, 100, 100] = 100
    t3[0, 0, 50, 100] = 100

    t4 = torch.zeros_like(t0)
    t4[0, 0, 100, 100] = 100
    t4[0, 0, 100, 50] = 100

    assert torch.isclose(loss.forward(t3, t4), torch.tensor(0.))
    assert torch.isclose(loss.forward(t4, t3), torch.tensor(0.))


def test_DiceLoss():
    loss = DiceLoss()

    t0 = torch.zeros(1, 1, 200, 200)
    assert torch.isclose(loss.forward(t0, t0), torch.tensor(0.))

    t1 = torch.zeros_like(t0)
    t1[0, 0, 100, 100] = 100
    assert torch.isclose(loss.forward(t0, t1), torch.tensor(1.))
    assert torch.isclose(loss.forward(t1, t0), torch.tensor(1.))

    t2 = torch.zeros_like(t0)
    t2[0, 0, 100, 100] = 100
    t2[0, 0, 50, 100] = 100
    t2[0, 0, 100, 50] = 100

    assert torch.isclose(loss.forward(t0, t2), torch.tensor(1.))
    assert torch.isclose(loss.forward(t2, t0), torch.tensor(1.))

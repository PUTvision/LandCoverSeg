import monai.losses
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

        self.loss = monai.losses.DiceLoss(
                include_background=False
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(inputs, targets)

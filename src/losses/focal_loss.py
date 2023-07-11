import monai.losses
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        
        weights = [0.34523039, 23.35764161,  0.60372157,  3.09578992, 12.32148783]

        self.loss = monai.losses.FocalLoss(
                include_background=True,
                weight=weights
            )
        

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(inputs, targets)

        return loss

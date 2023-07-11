import monai.losses
import torch
import torch.nn as nn


class FocalDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalDiceLoss, self).__init__()

        self.loss = monai.losses.DiceFocalLoss(
                include_background=True,
                reduction='none',
                softmax=True,
            )
        self.weights = torch.tensor([[0.34523039, 23.35764161,  0.60372157,  3.09578992, 12.32148783]]).cuda()


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(inputs, targets)

        loss = torch.mean(loss, dim=(2,3)) * self.weights
        return torch.mean(loss)

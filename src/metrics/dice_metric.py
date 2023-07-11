import torch
import torch.nn.functional as F
from torchmetrics import Metric
import monai.metrics


class DiceMetric(Metric):
    def __init__(self, classes: int, dist_sync_on_step: bool = False, smooth: float = 100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(DiceMetric, self).__init__()

        self._classes = classes
        self._smooth = smooth
        
        self.add_state('dice', default=torch.tensor([0.0 for _ in range(classes)]), dist_reduce_fx='sum')

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert inputs.shape == targets.shape
        
        inputs = torch.argmax(inputs, dim=1)
        inputs = F.one_hot(inputs, num_classes=self._classes)

        targets = torch.argmax(targets, dim=1)
        targets = F.one_hot(targets, num_classes=self._classes)
        
        targets = torch.flatten(targets, 1)
        inputs = torch.flatten(inputs, 1)
        intersection = torch.sum(targets * inputs, dim=1)
        dice = (2. * intersection + self._smooth) / (torch.sum(targets, dim=1) + torch.sum(inputs, dim=1) + self._smooth)

        self.dice += dice.mean(dim=0)

    def compute(self):
        return self.dice.mean().item()

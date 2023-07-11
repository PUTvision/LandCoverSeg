from typing import Optional, List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics

from src.losses.dice_loss import DiceLoss
from src.losses.focal_dice_loss import FocalDiceLoss
from src.losses.focal_loss import FocalLoss
from src.metrics.dice_metric import DiceMetric
from src.metrics.iou_metric import IOUMetric
from src.models.segmenter_visualizate_utils import visualize_segmentation_predition


class Segmenter(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 encoder_name: str,
                 input_channels: int,
                 classes: List[str],
                 loss_function: str,
                 lr: float,
                 lr_patience: int,
                 visualize_test_images: bool
                 ):
        super().__init__()

        self._model_name = model_name
        self._encoder_name = encoder_name
        self._input_channels = input_channels
        self._classes = classes
        self._loss_function = loss_function
        self._lr = lr
        self._lr_patience = lr_patience
        self._visualize_test_images = visualize_test_images

        if self._model_name == 'UNet':
            self.network = smp.Unet
        elif self._model_name == 'DeepLabV3Plus':
            self.network = smp.DeepLabV3Plus
        else:
            raise NotImplementedError(
                f'Unsupported model: {self._model_name}')

        self.network = self.network(
            encoder_name=self._encoder_name,
            encoder_weights="imagenet",
            in_channels=self._input_channels,
            classes=len(self._classes),
            activation=None
        )

        if loss_function == 'MAE':
            self.loss = torchmetrics.MeanSquaredError()
        elif loss_function == 'MSE':
            self.loss = torchmetrics.MeanSquaredError()
        elif loss_function == 'Dice':
            self.loss = DiceLoss()
        elif loss_function == 'Focal':
            self.loss = FocalLoss()
        elif loss_function == 'FocalDice':
            self.loss = FocalDiceLoss()
            
        else:
            raise NotImplementedError(
                f'Unsupported loss function: {loss_function}')

        metrics = torchmetrics.MetricCollection({
            'dice': DiceMetric(classes=len(self._classes)),
            'iou': IOUMetric(classes=len(self._classes)),
        })

        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        if torch.isinf(loss):
            return None

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics(y_pred, y))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(y_pred, y))

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log_dict(self.test_metrics(y_pred, y))

        if self._visualize_test_images and self.logger is not None:
            visualize_segmentation_predition(self.logger, x, y, y_pred)


    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.0005)
        
        self.s = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=237,
                eta_min=1e-7
            )

        sched = {
               'scheduler': self.s,
               'interval': 'step',
            }
        
        return [self.optimizer], [sched]

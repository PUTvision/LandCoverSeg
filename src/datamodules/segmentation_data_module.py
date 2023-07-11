import itertools
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, List, Tuple
from glob import glob 

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SegmentationDataModule(LightningDataModule):
    def __init__(self,
                 data_path: Path,
                 dataset: Dataset,
                 augment: bool,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 image_mean: Tuple[float, float, float],
                 image_std: Tuple[float, float, float],
                 number_of_workers: int,
                 number_of_splits: int,
                 current_split: int
                 ):
        super().__init__()

        self._data_root = Path(data_path)
        self._dataset = dataset
        self._augment = augment
        self._batch_size = batch_size
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std
        self._number_of_workers = number_of_workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            A.CenterCrop(self._image_size[0], self._image_size[1], always_apply=True),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ])

        self._augmentations = A.Compose([
            # rgb augmentations
            A.RandomGamma(gamma_limit=(80, 120)),
            A.ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5),
            A.ISONoise(color_shift=(0.01, 0.1)),
            # geometry augmentations
            A.Affine(rotate=(-5, 5), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.Flip(),
            # transforms
            A.RandomCrop(self._image_size[0], self._image_size[1], always_apply=True),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None):
        
        with open(str(self._data_root) + '/train.txt') as f:
            train_split = f.read().split('\n')[:-1]

        with open(str(self._data_root) + '/val.txt') as f:
            valid_split = f.read().split('\n')[:-1]

        with open(str(self._data_root) + '/test.txt') as f:
            test_split = f.read().split('\n')[:-1]

        self._train_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': train_split,
                'augmentations': self._augmentations if self._augment else self._transforms,
            })

        self._valid_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': valid_split,
                'augmentations': self._transforms,
            })

        self._test_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': test_split,
                'augmentations': self._transforms,
            })

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

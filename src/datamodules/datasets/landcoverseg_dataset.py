from typing import List, Tuple

from albumentations import Compose
import cv2
import numpy as np
from pathlib import Path
import torch
import zarr

from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    def __init__(self,
                 data_root: Path,
                 images_list: List,
                 augmentations: Compose
                 ):

        self._data_root = str(data_root)
        self._images_list = images_list
        self._augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_data(index)

        transformed = self._augmentations(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        return image, torch.multiply(mask.type(torch.float32), 1. / 255.).permute((2, 0, 1))

    def _load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_name = self._images_list[index]

        frame = cv2.imread(f'{self._data_root}/output/{image_name}.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label = cv2.imread(f'{self._data_root}/output/{image_name}_m.png', cv2.IMREAD_GRAYSCALE)

        mask = np.zeros((*label.shape[:2], 5), dtype=np.uint8)
        for i in range(5):
            mask[:, :, i][label == i] = 255
        #     cv2.imshow(f'mask_{i}', mask[:, :, i])
        # cv2.waitKey(0)

        return frame, mask

    def __len__(self) -> int:
        return len(self._images_list)

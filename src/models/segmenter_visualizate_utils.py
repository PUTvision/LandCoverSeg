from neptune.new.types import File
import torch
import random
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import cv2
import numpy as np


def get_palette(num_classes):
    # prepare and return palette
    # 'background', 'buildings', 'woodlands', 'water', 'roads'
        # palette = [
        #     0, 0, 0,
        #     255,0,0,
        #     0,255,0,
        #     0,0,255,
        #     255,255,0,
        # ]
        
        # 'background', 'building_flooded', 'building_non_flooded', 'road_flooded', 'road_non_flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'
        palette = [
            0,0,0,
            153,51,0,
            255,0,0,
            255,204,0,
            255,255,0,
            0,0,255,
            0,255,0,
            128,0,128,
            0,204,255,
            51,153,102
        ]

        return palette


def visualize_segmentation_predition(logger, x_s, y_s, y_pred_s, name='test_random_mask'):
    palette = get_palette(x_s.shape[1])

    for x, y, y_pred in zip(x_s, y_s, y_pred_s):

        x = np.uint8(x.permute(1, 2, 0).cpu().numpy()*255)
        y_pred = y_pred.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=0).astype(np.uint8)

        result_img = Image.fromarray(y_pred).convert('P', colors=palette)
        result_img.putpalette(palette)
        result_img = np.array(result_img.convert('RGB'))
        result_img = cv2.medianBlur(result_img, 5)

        frame = cv2.addWeighted(x, 0.3, result_img, 0.7, 0)
        
        logger.log_image(
                    name,
                    File.as_image(np.float32(frame)*1./255.)
                )

        break

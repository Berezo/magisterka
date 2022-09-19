# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:00:45 2022

@author: dynam
"""

from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img


class MapDataset(keras.utils.Sequence):
    

    def __init__(self, image_input_paths, image_mask_paths, batch_size):
        self.image_input_paths = image_input_paths
        self.image_mask_paths = image_mask_paths
        self.batch_size = batch_size
        self.image_size = (512, 512)

    def __len__(self):
        return len(self.image_mask_paths) // self.batch_size

    def __getitem__(self, index):
        """Returns tuple (input, mask) correspond to batch #index."""
        idx = index * self.batch_size
        batch_image_input_path = self.image_input_paths[idx : idx + self.batch_size]
        batch_image_mask_path = self.image_mask_paths[idx : idx + self.batch_size]
        x = load_img(batch_image_input_path[0], target_size=self.image_size)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255
        y = load_img(batch_image_mask_path[0], target_size=self.image_size)
        y = np.expand_dims(y, axis=0)
        y = y.astype('float32') / 255
        return x, y
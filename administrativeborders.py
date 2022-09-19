# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:00:45 2022

@author: dynam
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class AdministrativeBorders(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, image_size, image_input_paths, image_mask_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_input_paths = image_input_paths
        self.image_mask_paths = image_mask_paths

    def __len__(self):
        return len(self.image_mask_paths) // self.batch_size

    def __getitem__(self, index):
        """Returns tuple (input, mask) correspond to batch #index."""
        idx = index * self.batch_size
        batch_image_input_paths = self.image_input_paths[idx : idx + self.batch_size]
        batch_image_mask_paths = self.image_mask_paths[idx : idx + self.batch_size]
        x = np.zeros((self.batch_size,) + self.image_size + (3,), dtype="float32")
        for j, path in enumerate(batch_image_input_paths):
            image = load_img(path, target_size=self.image_size)
            x[j] = image
        y = np.zeros((self.batch_size,) + self.image_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_image_mask_paths):
            image = load_img(path, target_size=self.image_size, color_mode="grayscale")
            y[j] = np.expand_dims(image, 2)
            """ Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:"""
            y[j] -= 1
        return x, y
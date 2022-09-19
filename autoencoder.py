# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:19:30 2022

@author: dynam
"""

import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import PIL
from PIL import ImageOps

from administrativeborders import AdministrativeBorders

class Autoencoder:
    def __init__(self, image_size, num_classes):
        inputs = keras.Input(shape=image_size + (3,))
        
        ### [First half of the network: downsampling inputs] ###
    
        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    
        previous_block_activation = x  # Set aside residual
    
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
    
        ### [Second half of the network: upsampling inputs] ###
    
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.UpSampling2D(2)(x)
    
            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
    
        # Add a per-pixel classification layer
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    
        # Define the model
        self.model = keras.Model(inputs, outputs)
        self.image_size = image_size
        self.num_classes = num_classes

        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()


        self.model.summary()

    
    def validation_split(self, batch_size, image_input_paths, image_mask_paths):
        samples_val = 10
        random.Random(1500).shuffle(image_input_paths)
        random.Random(1500).shuffle(image_mask_paths)
        
        image_input_train_paths = image_input_paths[:-samples_val]
        image_mask_train_paths = image_mask_paths[:-samples_val]
        image_input_val_paths = image_input_paths[-samples_val:]
        image_mask_val_paths = image_mask_paths[-samples_val:]
        
        # print(len(image_input_train_paths))
        # print(len(image_mask_train_paths))
        # print(len(image_input_val_paths))
        # print(len(image_mask_val_paths))
        
        self.train_gen = AdministrativeBorders(batch_size - len(image_input_train_paths), self.image_size, image_input_train_paths, image_mask_train_paths)
        self.val_gen = AdministrativeBorders(batch_size - len(image_input_val_paths), self.image_size, image_input_val_paths, image_mask_val_paths)
        
        # print(len(self.train_gen))
    
    def train(self):
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

        callbacks = [
            keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
        ]

        # Train the model, doing validation at the end of each epoch.
        epochs = 10
        self.model.fit(self.train_gen, epochs=epochs, validation_data=self.val_gen, callbacks=callbacks)
    
    def generate_prediction(self):
        self.val_preds = self.model.predict(self.val_gen)
    
    def display_mask(self, i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(self.val_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        image = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        PIL.display(image)

autoencoder = Autoencoder((2048, 2048),1)
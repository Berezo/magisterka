# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:19:30 2022

@author: dynam
"""

import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model


from mapdataset import MapDataset
from matplotlib import pyplot as plt


class Autoencoder:
    def __init__(self, image_size):
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
         
        self.model.add(MaxPooling2D((2, 2), padding='same'))
             
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        
        self.model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
        
        self.model.summary()
    
    def validation_split(self, image_input_paths, image_mask_paths):
        batch_size = 1
        samples_valid = 5 #TODO zmiana liczby na 20% z ilosci obiektow
        seed = random.randint(1000,1500)
        random.Random(seed).shuffle(image_input_paths)
        random.Random(seed).shuffle(image_mask_paths)
        
        image_input_paths_train = image_input_paths[:-samples_valid]
        image_mask_paths_train = image_mask_paths[:-samples_valid]
        image_input_paths_valid = image_input_paths[-samples_valid:]
        image_mask_paths_valid = image_mask_paths[-samples_valid:]
        
        self.train_dataset = MapDataset(image_input_paths_train, image_mask_paths_train, batch_size)
        self.valid_dataset = MapDataset(image_input_paths_valid, image_mask_paths_valid, batch_size)
        
    
    def train(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  #Using binary cross entropy loss. Try other losses. 

        callbacks = [
            keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
        ]

        epochs = 10
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, callbacks=callbacks)
    
    def generate_prediction(self):
        self.predict_dataset = self.model.predict(self.valid_dataset)
    
    def display_mask(self, i):
        """Quick utility to display a model's prediction."""
        plt.figure(figsize=(20, 10))
        plt.subplot(1,3,1)
        plt.imshow(self.train_dataset[i][0][0,:,:,:])
        plt.title('Image')
        plt.subplot(1,3,2)
        plt.imshow(self.train_dataset[i][1][0,:,:,:])
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(self.predict_dataset[i,:,:,:])
        plt.title('Predicted Mask')
        plt.show()
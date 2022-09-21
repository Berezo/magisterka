# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:19:30 2022

@author: dynam
"""

import os
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
from osgeo import gdal


from mapdataset import MapDataset



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
    
    def validation_split(self, image_input_paths, image_mask_paths): #TODO stworzenie osobnego datasetu predykcyjnego
        batch_size = 1
        samples_valid = 5 #TODO zmiana liczby na 20% z ilosci obiektow
        seed = random.randint(1000,1500)
        random.Random(seed).shuffle(image_input_paths)
        random.Random(seed).shuffle(image_mask_paths)
        
        image_input_paths_train = image_input_paths[:-samples_valid]
        image_mask_paths_train = image_mask_paths[:-samples_valid]
        self.image_input_paths_valid = image_input_paths[-samples_valid:]
        image_mask_paths_valid = image_mask_paths[-samples_valid:]
        
        self.train_dataset = MapDataset(image_input_paths_train, image_mask_paths_train, batch_size)
        self.valid_dataset = MapDataset(self.image_input_paths_valid, image_mask_paths_valid, batch_size)
        
    def image_name_list(self, path):
        return sorted(
            [
                os.path.join(path, fname[fname.index('M'):]).replace('\\', '/').replace('.JPG', '.tif')
                for fname in self.valid_dataset.image_input_paths
                if fname.endswith(".JPG")
            ]
        )
        
    
    def train(self):#TODO Dodać Epoch jako zmienna
        epochs = 1
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  #Using binary cross entropy loss. Try other losses. 

        callbacks = [
            keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
        ]
        
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, callbacks=callbacks)
    
    def generate_prediction(self):  #TODO stworzenie osobnego datasetu predykcyjnego
        self.predict_dataset = self.model.predict(self.valid_dataset)
    
    def display_mask(self, i): 
        """Quick utility to display a model's prediction."""
        plt.figure(figsize=(20, 10))
        plt.subplot(1,3,1)
        plt.imshow(self.valid_dataset[i][0][0,:,:,:])
        plt.title('Image')
        plt.subplot(1,3,2)
        plt.imshow(self.valid_dataset[i][1][0,:,:,:])
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(self.predict_dataset[i,:,:,:])
        plt.title('Predicted Mask')
        plt.show()
    
    def save_prediction(self, path):
        image_names = self.image_name_list(path)
        for index, prediction in enumerate(self.predict_dataset):
            keras.preprocessing.image.save_img(image_names[index], prediction)
            print('Saved {} from {} predictions.'.format(index+1, len(image_names)))
    
    def copy_spatial_reference(self, path): #TODO działa ale to pierwotny przypadek
        image_names = self.image_name_list(path)
        for index, prediction in enumerate(image_names):
            raster_input = gdal.Open(self.image_input_paths_valid[index], gdal.GA_ReadOnly)
            projection = raster_input.GetProjection()
            geotransform = raster_input.GetGeoTransform()
            del raster_input

            raster_prediction = gdal.Open(prediction, gdal.GA_Update)
            raster_prediction.SetGeoTransform(geotransform)
            raster_prediction.SetProjection(projection)
            print('Copy spatial reference {} from {} predictions.'.format(index+1, len(image_names)))
            del raster_prediction
        

        
        
        
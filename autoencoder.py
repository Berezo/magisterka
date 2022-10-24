# -*- coding: utf-8 -*-

import os, random, math
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K
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
        samples_number = len(image_input_paths)
        samples_valid = math.floor(0.2 * samples_number)
        test_valid = math.floor(0.1 * samples_number)
        
        seed = random.randint(1000,1500)
        random.Random(seed).shuffle(image_input_paths)
        random.Random(seed).shuffle(image_mask_paths)
        
        image_input_paths_train = image_input_paths[:-samples_valid-test_valid]
        image_mask_paths_train = image_mask_paths[:-samples_valid-test_valid]
        image_input_paths_test = image_input_paths[-samples_valid-test_valid:-samples_valid]
        image_mask_paths_test = image_mask_paths[-samples_valid-test_valid:-samples_valid]
        image_input_paths_valid = image_input_paths[-samples_valid:]
        image_mask_paths_valid = image_mask_paths[-samples_valid:]
        self.train_dataset = MapDataset(image_input_paths_train, image_mask_paths_train, batch_size)
        self.valid_dataset = MapDataset(image_input_paths_valid, image_mask_paths_valid, batch_size)
        self.test_dataset = MapDataset(image_input_paths_test, image_mask_paths_test, batch_size)
        
    def image_name_list(self, image_input_paths, path_output):
        return sorted(
            [
                os.path.join(path_output, fname[fname.index('M'):]).replace('\\', '/').replace('.JPG', '.jpg')
                for fname in image_input_paths
                if fname.endswith(".JPG")
            ]
        )
        
    def get_f1(self, y_true, y_pred): 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    
    def get_accuracy_plot(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
    def train(self, epochs):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',
                           'Precision', 'Recall', self.get_f1, MeanIoU(num_classes=6)])

        callbacks = [
            keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
        ]
        
        history = self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, callbacks=callbacks)
        
        self.get_accuracy_plot(history)
        
        self.model.evaluate(self.test_dataset, callbacks=callbacks)
    
    def generate_prediction(self, image_input_paths):  #TODO stworzenie osobnego datasetu predykcyjnego
        batch_size = 1
        to_predict_dataset = MapDataset(image_input_paths, image_input_paths, batch_size)
    
        self.predict_dataset = self.model.predict(to_predict_dataset)
    
    def save_prediction(self, image_input_paths, path_output):
        image_names = self.image_name_list(image_input_paths, path_output)
        for index, prediction in enumerate(self.predict_dataset):
            keras.preprocessing.image.save_img(image_names[index], prediction)
        print('Saved predictions')
    
    def copy_spatial_reference(self, image_input_paths, path_output): #TODO działa ale to pierwotny przypadek
        image_names = self.image_name_list(image_input_paths, path_output)
        for index, prediction in enumerate(image_names):
            raster_input = gdal.Open(image_input_paths[index], gdal.GA_ReadOnly)
            projection = raster_input.GetProjection()
            geotransform = raster_input.GetGeoTransform()
            del raster_input

            raster_prediction = gdal.Open(prediction, gdal.GA_Update)
            raster_prediction.SetGeoTransform(geotransform)
            raster_prediction.SetProjection(projection)
            del raster_prediction
        print('Copy spatial reference to predictions.')
        

        
        
        
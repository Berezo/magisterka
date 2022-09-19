# -*- coding: utf-8 -*-
"""
Color detection test
"""
from administrativeborders import AdministrativeBorders
from autoencoder import Autoencoder
import tensorflow as tf
import os

import numpy as np
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from matplotlib import pyplot as plt


def main():
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # if tf.config.list_physical_devices('GPU'):
    #     physical_devices = tf.config.list_physical_devices('GPU')
    #     tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    #     tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    
    # input_dir = "./input"
    # mask_dir = "./mask"
    
    # image_input_paths = sorted(
    #     [
    #         os.path.join(input_dir, fname).replace('\\', '/')
    #         for fname in os.listdir(input_dir)
    #         if fname.endswith(".JPG")
    #     ]
    # )
    # image_mask_paths = sorted(
    #     [
    #         os.path.join(mask_dir, fname).replace('\\', '/')
    #         for fname in os.listdir(mask_dir)
    #         if fname.endswith(".JPG") and not fname.startswith(".")
    #     ]
    # )
    # image_size = (512, 512)
    # batch_size = 20
    # num_classes = 1
    
    # administrativeborders = AdministrativeBorders(batch_size, image_size, image_input_paths, image_mask_paths)
    # autoencoder = Autoencoder(image_size, num_classes)
    
    # autoencoder.validation_split(batch_size, image_input_paths, image_mask_paths)
    # autoencoder.train()
    # autoencoder.generate_prediction()
    # autoencoder.display_mask(3)
    
    image_size = 512
    
    image = cv2.imread('C:/Studia/magisterka/git/magisterka/input/MARP_25_RADOM_1937_0.JPG')
    print(image.shape)
    
    image_array = np.expand_dims(image, axis=0)
    print(image_array.shape)
    image_array = image_array.astype('float32') / 255
    
    
    mask = cv2.imread('C:/Studia/magisterka/git/magisterka/mask/MARP_25_RADOM_1937_0_MASK.JPG')
    print(mask.shape)
    
    mask_array = np.expand_dims(image, axis=0)
    print(mask_array.shape)
    mask_array = mask_array.astype('float32') / 255
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
     
    
    model.add(MaxPooling2D((2, 2), padding='same'))
         
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  #Using binary cross entropy loss. Try other losses. 
    model.summary()
    
    model.fit(image_array, mask_array, epochs=500)

    prediction = model.predict(image_array)
    print(prediction.shape)
    print(prediction.max())
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(prediction[0,:,:,:])
    plt.title('Predicted Mask')
    plt.show()
    

if __name__ == "__main__":
    main()
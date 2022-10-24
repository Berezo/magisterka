# -*- coding: utf-8 -*-
"""
Main
"""

from autoencoder import Autoencoder
import os

def main():
    image_size = 512
    input_dir = "./input" 
    input_predict_dir = "./input_predict" 
    mask_dir = "./mask" 
    predict_dir = "./predict" 
    
    image_input_paths = sorted(
        [
            os.path.join(input_dir, fname).replace('\\', '/')
            for fname in os.listdir(input_dir)
            if fname.endswith(".JPG")
        ]
    )
    
    image_mask_paths = sorted(
        [
            os.path.join(mask_dir, fname).replace('\\', '/')
            for fname in os.listdir(mask_dir)
            if fname.endswith(".JPG")  and not fname.startswith(".")
        ]
    )

    image_input_to_predict_paths = sorted(
        [
            os.path.join(input_predict_dir, fname).replace('\\', '/')
            for fname in os.listdir(input_predict_dir)
            if fname.endswith(".JPG")
        ]
    )

    model = Autoencoder(image_size)    
    model.validation_split(image_input_paths, image_mask_paths)
    model.train(10)
    model.generate_prediction(image_input_to_predict_paths)
    model.save_prediction(image_input_to_predict_paths, predict_dir)
    model.copy_spatial_reference(image_input_to_predict_paths, predict_dir)
    
    
if __name__ == "__main__":
    main()
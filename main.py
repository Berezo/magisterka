# -*- coding: utf-8 -*-
"""
Main
"""

from autoencoder import Autoencoder
import os

def main():
    image_size = 512
    input_dir = "./input"
    mask_dir = "./mask"
    
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
            if fname.endswith(".JPG") and not fname.startswith(".")
        ]
    )

    model = Autoencoder(image_size)    
    model.validation_split(image_input_paths, image_mask_paths)
    model.train()
    model.generate_prediction()
    model.display_mask(0)
    
if __name__ == "__main__":
    main()
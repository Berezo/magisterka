# -*- coding: utf-8 -*-
"""
Mask creation
"""
import colordetector
import cv2
import os

def check_if_path_exist(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def list_image_paths(path):
    return sorted(
        [
            os.path.join(path, fname).replace('\\', '/')
            for fname in os.listdir(path)
            if fname.endswith(".JPG")
        ]
    )

def list_image_names(path):
    return sorted(
        [
            fname
            for fname in os.listdir(path)
            if fname.endswith(".JPG")
        ]
    )

def main():
    path_input = "./input"
    path_mask = "./mask"
    path_canny = "./canny"
    color_detector = colordetector.ColorDetector()
    
    check_if_path_exist(path_mask)
    check_if_path_exist(path_canny)
    check_if_path_exist(path_input)
    
    image_paths = list_image_paths(path_input)
    image_names = list_image_names(path_input)
    
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        
        mask = color_detector.detect_mask(image)
        cv2.imwrite(os.path.join(path_mask, image_names[index]).replace('\\', '/'), mask)
        
        canny = cv2.Canny(mask,100,150)
        cv2.imwrite(os.path.join(path_canny, image_names[index]).replace('\\', '/'), canny)
        
        print('Zapisano {} na {} masek'.format(index+1, len(image_paths)))

if __name__ == "__main__":
    main()


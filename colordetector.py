# -*- coding: utf-8 -*-
"""
Color detection
"""
import cv2
import numpy as np


class ColorDetector:
    lower_red_mask_1 = np.array([0, 50, 70])
    upper_red_mask_1 = np.array([10, 255, 255])
    lower_red_mask_2 = np.array([155,25,0])
    upper_red_mask_2 = np.array([180,255,255])     
    
    def detect_mask(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_1 = cv2.inRange(image_hsv, self.lower_red_mask_1, self.upper_red_mask_1)
        mask_2 = cv2.inRange(image_hsv, self.lower_red_mask_2, self.upper_red_mask_2)
        return mask_1 + mask_2
    
    def detect_result(self, image):
        mask = self.detect_mask(image)
        image_result = cv2.bitwise_and(image, image, mask=mask)
        return image_result
        

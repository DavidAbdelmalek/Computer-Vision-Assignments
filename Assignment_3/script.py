#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# David Onsy Shokry                 34-4658         t-08
# Anas Mohammed Amin                34-10441        t-08



import cv2
import numpy as np
import math
from scipy import ndimage
from PIL import Image

def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    #left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    #right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w_l, h,_ = left_img.shape  # both images have different number of rows so apply min aggregation.
    
    w_r,_,_ = right_img.shape # both have the same height.

    w=0
    w = min(w_r,w_l)
    
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
    
    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
    
    for y in range(kernel_half, h - kernel_half):
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0
                
                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(left[y+v, x+u,0]) - int(right[y+v, (x+u) - offset,0])
                        ssd += ssd_temp * ssd_temp
                
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
    
        # set depth output for this x,y location to the best match
        depth[y, x] = best_offset * offset_adjust


def main():
    image = cv2.imread("As3.jpg")
    cv2.imshow('image',image)
    
    left_img =cv2.imread("As3_left.jpg")
    right_img = cv2.imread("As3_right.jpg")
    
    
    stereo_match(left_img, right_img, 7, 75)
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage


# David Onsy Shokry                 34-4658
# Anas Mohammed Amin                34-10441

#   import matplotlib.ppylot as plt

path = "imagesA2"
imgs = ["","/1.jpg","/2.jpg","/3.jpg","/4.jpg","/5.jpg","/6.jpg","/7.jpg","/8.jpg"]


def winVar(img, wlen):
  wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
    borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
  return wsqrmean - wmean*wmean


def noise(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('Image', img)
    rows, cols, channels = img.shape
    noise=0
    var = 10000
    num_pixels = rows*cols
    for i in range(rows):
        for j in range(cols):
            cum = 0
            sum =0
            pixels=0
            for x in range(i-1,i+2,1):
                for y in range(j-1,j+2,1):
                    if(x>0 and y>0 and x<rows and y<cols):
                        sum = sum + img[x][y]
                        pixels = pixels+1

            mean  = sum/pixels
            variance= (img[i][j]-mean)**2
            if (variance[0]>var):
                noise = noise+1

    noise_percentage = (noise/num_pixels)*100
    # Detecting noise
    if (noise_percentage > 4):
        print("Noise percentage: {:.2f}".format(noise_percentage))
        img_median = cv2.medianBlur(img,5)
        cv2.imshow("Median Blur on Image",img_median)
    else:
        print("The image has now noise")



def blur(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('Image', img)
    rows, cols, _ = img.shape
    

    scale = 1
    delta = 0
    
    sobelx = cv2.Sobel(img,cv2.CV_32F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_32F,0,1)

    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    G = abs_sobelx+abs_sobely

    blur_pixels = 0
    blur_prec = 0
    for i in range(rows):
        for j in range(cols):
            intenisty=G[i][j][0]
            if(intenisty>150):
                blur_pixels = blur_pixels+1


    pixels = rows*cols
    blur_prec = (blur_pixels / pixels) * 100;

    print("Number of blurry pixels are {} pixels, with precentage {:.2f} %".format(blur_pixels,blur_prec))

    if(blur_prec<7):
        print("Since it is less than 7% so it requires filter")
        blur = cv2.GaussianBlur(img,(99,99),0)
        sharp_image = cv2.addWeighted(img,1.7,blur,-0.7,0)
        cv2.imshow("Filtered Image 7", sharp_image)
    else:
        print("No Blurry")


#print("-"*100)
#print(G[:,:,1])

def histogram_equalization(img,freq,cum_histogram,min_intensity,max_intensity):
    f_min =cum_histogram[min_intensity]
    new_img = img
    rows, cols, _ = img.shape
    new_cum_histogram =np.zeros((1,256),dtype= int)
    num_pixels = cols *rows
    
    for i in range(min_intensity,max_intensity+1):
        fr = cum_histogram[i]- f_min
        sec = 255/(num_pixels-f_min)
        new_cum_histogram[i]=fr*sec


    return new_cum_histogram


def color_collapsing(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('Image', img)
    rows, cols, _ = img.shape
    
    freq= np.zeros((1,256),dtype= int)
    

    min_intensity=1000000000
    max_intensity=0
    

# looping over the whole image to detect the freq of each intenisty in the image.
    for i in range(rows):
        for j in range(cols):
            intenisty=img[i][j][0]
            freq[0,intenisty]=freq[0,intenisty]+1
            # check for the min and max_freq
            if(intenisty>max_intensity):
                max_intensity= intenisty
            if(intenisty<min_intensity):
                min_intensity=intenisty

    cum_histogram= np.zeros((1,256),dtype= int)
    cum_histogram[0,0]=freq[0,0]
    for i in range(1,max_intensity+1):
        cum_histogram[0,i]=cum_histogram[0,i-1]+freq[0,i]
    
    print("Max intensity for this image is {} with frequency {}".format(max_intensity,freq[0,max_intensity]))
    print("Min intensity for this image is {} with frequency {}".format(min_intensity,freq[0,min_intensity]))

    range_intensities = 100 *((max_intensity-min_intensity)/255)
    print("Range of intensities is {:.2f}%".format(range_intensities))

    c = 0
    d= 0
    min_found= False
    num_pixels=cols*rows
    for i in range(1,max_intensity+1):
        if(not(min_found)):
            if(cum_histogram[0,i]>=(5/100)*num_pixels):
                min_found=True
                c = i
        if(cum_histogram[0,i]>=(95/100)*num_pixels):
            d = i
            break

    if(range_intensities<50):
        print("There is color collapsing")

        a = 0
        b = 255
        div =(b-a)/(d-c)
        
        new_img = img
        for i in range(rows):
            for j in range(cols):
        
                new_intenisty= int((((img[i][j][0]-c) * div) + a))
                if(new_intenisty>255):
                    new_intenisty = 255
                new_img[i][j][0]=new_intenisty
                new_img[i][j][1]=new_intenisty
                new_img[i][j][2]=new_intenisty



        cv2.imshow('Color Stretched', new_img)
    else:
        print("No color collapsing in this image")



def main(index):
    

    #noise(path+imgs[index])
    #blur(path+imgs[index])
    color_collapsing(path+imgs[index])
    

    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    index = int(input("Please insert the number of image you want to test over\n"))
    main(index)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from scipy import ndimage

#   import matplotlib.ppylot as plt

path = "/Users/dodo/GUC/sem_9/Computer_Vision/Assignments/Assignment_1/A1I"
#file_to_save = "/Users/dodo/GUC/sem_9/Computer_Vision/Assignments/Assignment_1/out_imgs"

def edit_pic(imgpath):
    
    # Read the image
    img = cv2.imread(imgpath, 1)
    
    # Get the shape of the image.
    rows, cols, channels = img.shape
    
    alpha=10
    i=0
    for i in range(rows):
        alpha =2.42
        for j in range(cols):
            if(j>350):
                alpha-=0.003
            for c in range(3):
                img[i,j,c]=alpha*img[i,j,c]+ 5

    return img

def rotateImage(imgpath, angle):
    image =cv2.imread(imgpath, 1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def geom_editing(imgpath):
    img = cv2.imread(imgpath, 1)
    rows, cols, channels = img.shape
    
    # Rotating 180degree around the origin
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    edited = cv2.warpAffine(img, M, (cols, rows),flags=cv2.INTER_LINEAR)
   
    # flipping around x-axis
    im2 = cv2.flip(edited, 0)
    # Resizing image to have the same shape of the image Q10.
    resized_image = cv2.resize(im2, (1121, 788))
    
    
    res=  rotateImage(imgpath,180)
    im2 = cv2.flip(res, 0)
    

    translation_matrix = np.float32([ [1,0,135], [0,1,0] ])
    img_translation = cv2.warpAffine(im2, translation_matrix, (cols, rows))
    return img_translation





def rotateImage2(image, angle):
    rows,cols,_ = image.shape
    rot_mat = cv2.getRotationMatrix2D(((cols/2,rows/2)), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (cols,rows))
    return result


def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthB),int(widthA))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    
    dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
        
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                    
    # return the warped image
    return warped , maxWidth,maxHeight


def getDimension(rect):
    """It takes the rectangle dimension by any order of four points
       
        Returns the maxWidth and maxHeight, which are the real dimensions of Rectangle.
        """
    
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthB),int(widthA))
 
 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    return maxWidth,maxHeight






def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    
    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))
    
    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
    
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat




def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
        
        Crops blank image to 1x1.
        
        Returns cropped image.
        
        """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    
    return image

def main():
    
    #################################### Task 1 ####################################
    img1 = edit_pic(path +"/Q1I1.png")
    img2 =  geom_editing(path+"/Q1I2.jpg")
    cv2.imshow('img2',img2)
    rows,columns,channels = img1.shape
    img2 = cv2.resize(img2, (1121,788 ),cv2.INTER_LINEAR)
    
    alpha= 1
    beta = ( 1.0 - alpha );
    res =cv2.addWeighted( img1, 1, img2, 0.3, 0.0)
    
    cv2.imshow('img1',res)
    
    
    
    
    #################################### Task 2 ####################################
  
    img1 = cv2.imread(path + "/Q2I2.jpg")
    img2 = cv2.imread(path + "/Q2I1.jpg")
    resized_image = cv2.resize(img2, (91, 141),cv2.INTER_LINEAR)
    
    y_offset = 377
    x_offset = 1219
    img1[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image
    
    #cv2.imshow('output3',img1)

    ###########################                                 ###########################

    img = cv2.imread(path+"/Q2I3.jpg")
    img2 = cv2.imread(path + "/Q2I1.jpg")
    rows,columns , channels = img.shape
    boom  = rotate_image(img,6)
    
    pt_old = np.float32([[371,93],[670,127],[664,558],[325,525]])
    maxWidth,maxHeight = getDimension(pt_old)
    img2 = cv2.resize(img2, (maxWidth,maxHeight ),cv2.INTER_LINEAR)
   
    y_offset = 160
    x_offset = 380
    
    boom[y_offset:y_offset+img2.shape[0] , x_offset:x_offset+img2.shape[1]] = img2

    imr = rotate_image(boom,-6)
    imr = autocrop(imr)

    output4 = cv2.resize(imr, (columns,rows ),cv2.INTER_LINEAR)
    rows,columns , channels = imr.shape
    #cv2.imshow('output4',output4)

    
    #################################### Task 3 ####################################
    
    
    img = cv2.imread(path+"/Q3I1.jpg")
    img2 = cv2.imread(path + "/Q2I1.jpg")
    
    rows,columns,channels=img.shape
    rows2,columns2,channels=img2.shape
    
    pt_new = np.float32([[162,35],[464,70],[463,352],[158,388]])
    pt_old = np.float32([[0,0],[rows2,0],[rows2,columns2],[0,columns2]])
    
    #Getting the size of what is inside the frame and then resizing the shelock image.
    coordiates = np.float32([[164,37],[464,70],[463,352],[158,388]])
    #coordiates= np.float32([[164,37],[475,75],[470,358],[165,395]])

    image,_,_ = four_point_transform(img,coordiates)
    img2_resized= cv2.resize(img2, (710,500),cv2.INTER_LINEAR)
    
    
    #Rotating the image back
    rows_origin,columns_origin,_ = img.shape
    
    M = cv2.getPerspectiveTransform(pt_old, pt_new)
    warped = cv2.warpPerspective(img2_resized, M, (columns_origin, rows_origin))
    

    
    result = img+warped
    #cv2.imshow('output5',result)
    

    
    
    
    
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


import cv2
import numpy as np

def gradient(img):

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx64 = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=3)
    sobely64 = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=3)

    sobelx = np.uint8(np.absolute(sobelx64))
    sobely = np.uint8(np.absolute(sobely64))
    height,width= img.shape[:2]
    grad=0.5*(sobelx+sobely)
    return grad
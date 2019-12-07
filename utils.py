import cv2
import numpy as np
import random

def gradient(img):

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx64 = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=3)
    sobely64 = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=3)

    sobelx = np.uint8(np.absolute(sobelx64))
    sobely = np.uint8(np.absolute(sobely64))

    grad=(sobelx+sobely)//2
    return grad

def visualisation(A_padding,C,FNN,taille,holes_coord):
    img=C.copy()
    heightA_padding,widthA_padding = A_padding.shape[:2]
    heightA,widthA = heightA_padding-2*taille,widthA_padding-2*taille

    for x in range(taille, widthA+taille):
        if(x%10==0):
            for y in range(taille, heightA+taille):
                if(y%10==0):
                    x_ = holes_coord['x_min']+x-taille
                    y_ = holes_coord['y_min']+x-taille
                    start_point=(x_,y_)
                    end_point=(FNN[y][x][1],FNN[y][x][0])
                    R=random.randint(0,255)
                    G=random.randint(0,255)
                    B=random.randint(0,255)
                    cv2.line(img,start_point,end_point,(B,G,R),1)
    return img

def verification(A_padding,C,FNN,taille,holes_coord):
    img=C.copy()
    heightA_padding,widthA_padding = A_padding.shape[:2]
    heightA,widthA = heightA_padding-2*taille,widthA_padding-2*taille
    print("coord du trou ")
    print(holes_coord)
    for x in range(taille, widthA+taille):
            for y in range(taille, heightA+taille):
                    xB=FNN[y][x][1]
                    yB=FNN[y][x][0]
                    if(holes_coord['x_min']<=xB<=holes_coord['x_max'] and holes_coord['y_min']<=yB<=holes_coord['y_max']):
                        print(f"ERREUR : {y},{x} matche avec {FNN[y][x]}")
    print("coord du trou ")
    print(holes_coord)
    print("PAS D'ERREUR")
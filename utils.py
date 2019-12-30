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

def visualisation(A,FNN,taille,holes_coord,He):
    img=A.copy()
    heightA,widthA = A.shape[:2]
    x_min = holes_coord["x_min"]
    x_max = holes_coord["x_max"]
    y_min = holes_coord["y_min"]
    y_max = holes_coord["y_max"]
    for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if He[y,x]==1:
                    start_point=(x,y)
                    end_point=(FNN[y][x][1],FNN[y][x][0])
                    R=random.randint(0,255)
                    G=random.randint(0,255)
                    B=random.randint(0,255)
                    cv2.line(img,start_point,end_point,(B,G,R),1)
    return img

def verification(A,FNN,taille,holes_coord,He):
    x_min = holes_coord["x_min"]
    x_max = holes_coord["x_max"]
    y_min = holes_coord["y_min"]
    y_max = holes_coord["y_max"]
    print("coord du trou ")
    print(holes_coord)
    for x in range(x_min,x_max):
            for y in range(y_min,y_max):
                if He[y,x]==1:
                    xB=FNN[y][x][1]
                    yB=FNN[y][x][0]
                    if(holes_coord['x_min']<=xB<holes_coord['x_max'] and holes_coord['y_min']<=yB<holes_coord['y_max']):
                        print(f"ERREUR : {y},{x} matche avec {FNN[y][x]}")
    print("coord du trou ")
    print(holes_coord)
    print("PAS D'ERREUR")

def evaluation(img_origine,img_completee):
    return cv2.norm(img_origine,img_completee)
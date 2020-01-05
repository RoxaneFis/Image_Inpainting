import cv2
import argparse
import numpy as np
import jupyterlab
import datetime
from time import strftime
from patchmatch import initialisation, propagation, random_search
from utils import gradient, visualisation, verification, evaluation
from trackbar import get_hole, paint_draw, parameters


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    parser.add_argument('--nombre_etapes', type=int, help='nombre d etapes', default=9)
    parser.add_argument('--taille', type=int, help='taille du patch',default=10)
    parser.add_argument('--scale', type=int, help='Echelle random search', default=20)
    parser.add_argument('--alpha', type=int, help='Coefficient pour le gradient', default=0.7)
    parser.add_argument('--sizebrush', type=int, help='Taille de la brosse', default=15)
    parser.add_argument('--dir_name', type=str, help='Repertoire où stocker les images', default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':  
    #HYPERPARAMETERS
    args = parse_args()
    taille=args.taille
    scale=args.scale
    alpha = args.alpha
    nombre_etape=args.nombre_etapes
    size_brush = args.sizebrush
    dir_name = args.dir_name

    #IMAGE A COMPLETER
    B =cv2.imread(args.image_input)
    B_origine = B.copy()
    heightB,widthB = B.shape[:2]
    cv2.rectangle(B,(taille,taille),(widthB-taille,heightB-taille),(233))
    cv2.imshow("image_B",B)
    grad = gradient(B)
    cv2.imshow("gradient",grad)


    #INITIALISATION TROU
    cv2.imshow("Windows",B)
    param = get_hole(B,size_brush,"Windows")
    holes_coord = {"x_min":param.x_min,"x_max":param.x_max,"y_min":param.y_min,"y_max":param.y_max}
    He =param.hole
    for x in range(param.width-1):
        for y in range(param.height-1):
            if He[y,x]==1:
                B[y,x]=(0,0,0)
    cv2.imshow("B_painted",B)
    cv2.imwrite(f"{dir_name}image_trouee.jpg",B)
   
    #ALGORITHME
        #A.PATCH STATIQUE
    FNN, A = initialisation(He,B,taille,holes_coord)
    cv2.imshow("image_A_ini",A)
    cv2.imwrite(f"{dir_name}image_A_ini.jpg",A)
    for etape in range(nombre_etape):
        #Propagation
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
        cv2.imshow(f"{etape}_propagation_taille_{taille}.jpg",A)
        cv2.imwrite(f"{dir_name}_static_{etape}_propagation_taille_{taille}.jpg",A)
        #Random Search
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
        cv2.imshow(f"{etape}_random_taille_{taille}.jpg",A)
        cv2.imwrite(f"{dir_name}_random_{etape}_random_taille_{taille}.jpg",A)

        #B.PATCH DYNAMIQUE
    FNN, A = initialisation(He,B,taille,holes_coord)
    for etape in range(nombre_etape):
        #Propagation
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
        cv2.imshow(f"{etape}_propagation_taille_{taille}.jpg",A)
        cv2.imwrite(f"{dir_name}_dynamic_{etape}_propagation_taille_{taille}.jpg",A)
        #Random Search
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
        cv2.imshow(f"{etape}_random_taille_{taille}.jpg",A)
        cv2.imwrite(f"{dir_name}_dynamic_{etape}_random_taille_{taille}.jpg",A)
        taille = taille-1
    cv2.waitKey()



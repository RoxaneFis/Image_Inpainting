import cv2
import argparse
import numpy as np
import jupyterlab
import datetime
from time import strftime
from patchmatch import initialisation, propagation, random_search
from utils import gradient, visualisation, verification
from trackbar import get_hole, paint_draw, parameters


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    parser.add_argument('--nombre_etapes', type=int, help='nombre d etapes', default=3)
    parser.add_argument('--taille', type=int, help='taille du patch',default=20)
    parser.add_argument('--scale', type=int, help='Echelle random search', default=100)
    parser.add_argument('--alpha', type=int, help='Coefficient pour le gradient', default=0.7)
    parser.add_argument('--sizebrush', type=int, help='Taille de la brosse', default=15)
    parser.add_argument('--dir_name', type=str, help='Repertoire où stocker les images', default="")
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("Debut")
    
    #HYPERPARAMETERS
    args = parse_args()
    taille=args.taille
    scale=args.scale
    alpha = args.alpha
    nombre_etape=args.nombre_etapes
    size_brush = args.sizebrush
    dir_name = args.dir_name

    #INITIAL IMAGE
    B =cv2.imread(args.image_input)
    heightB,widthB = B.shape[:2]
    cv2.rectangle(B,(taille,taille),(widthB-taille,heightB-taille),(233))
    cv2.imshow("image_B",B)
    grad = gradient(B)
    cv2.imshow("gradient",grad)


    #INITIALIZE HOLE
    name = "Windows"
    cv2.imshow(name,B)
    param = get_hole(B,size_brush,name)
    holes_coord = {"x_min":param.x_min,"x_max":param.x_max,"y_min":param.y_min,"y_max":param.y_max}
    He =param.hole
    for x in range(param.width-1):
        for y in range(param.height-1):
            if He[y,x]==1:
                B[y,x]=(0,0,0)
    cv2.imshow("B_painted",B)
    date = datetime.datetime.now()
    heure = date.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    name =f"{heure}_image_trouee.jpg"
    cv2.imwrite(dir_name+name,B)
    #INITIALISATION

    print(heure)
    FNN, A = initialisation(He,B,taille,holes_coord)
    cv2.imshow("image_A_ini",A)
    name =f"{heure}_image_A_ini.jpg"
    cv2.imwrite(dir_name+name,A)
    #visu=visualisation(A_padding,C,FNN,taille,holes_coord)
    #cv2.imshow(f"Visu_propagation_ini",visu)
    for etape in range(nombre_etape):
        taille = taille-1
        # taille = tailles[etape%6]
        print(f"Phase de propagation: {etape} taille: {taille}")
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
        if (etape%1==0): 
            name = f"{heure}_A_etape_{etape}_propagation_taille_{taille}.jpg"
            cv2.imshow(name,A)
            cv2.imwrite(dir_name+name,A)
            #visu=visualisation(A, FNN,taille,holes_coord, He)
            #cv2.imshow(f"Visu_propagation_{etape}",visu)
        
        print(f"Random search {etape} taille: {taille}")
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
        if (etape%1==0): 
            name = f"{heure}_A_etape_{etape}_random_taille_{taille}.jpg"
            cv2.imshow(name,A)
            print(dir_name+name)
            cv2.imwrite(dir_name+name,A)
            #visu=visualisation(A,FNN,taille,holes_coord, He)
            #cv2.imshow(f"Visu_random_{etape}",visu)
    cv2.waitKey()



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
    parser.add_argument('--nombre_etapes', type=int, help='nombre d etapes', default=3)
    parser.add_argument('--taille', type=int, help='taille du patch',default=21)
    parser.add_argument('--scale', type=int, help='Echelle random search', default=100)
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
    nombre_etape=args.nombre_etapes
    dir_name = args.dir_name

    #INITIAL IMAGE
    B =cv2.imread(args.image_input)
    B_origine=B.copy()
    print(evaluation(B_origine,B))
    heightB,widthB = B.shape[:2]
    cv2.rectangle(B,(taille,taille),(widthB-taille,heightB-taille),(233))
    cv2.imshow("image_B",B)
    grad = gradient(B)
    cv2.imshow("gradient",grad)
    height,width = B.shape[:2]


    #INITIALIZE HOLE
    xmin=width//2-80
    xmax=xmin+120
    ymin=height//2+40
    ymax=ymin+120
    holes_coord = {"x_min":xmin,"x_max":xmax,"y_min":ymin,"y_max":ymax}
    He =np.zeros((height,width),dtype='i')
    for x in range(xmin,xmax):
        for y in range(ymin,ymax):
            He[y][x]=1
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
    print(f"Evaluation distance image origine initialisation : {evaluation(B_origine,A)}")
    #visu=visualisation(A_padding,C,FNN,taille,holes_coord)
    #cv2.imshow(f"Visu_propagation_ini",visu)
    for etape in range(nombre_etape):
        #taille = taille-1
        # taille = tailles[etape%6]
        print(f"Phase de propagation: {etape} taille: {taille}")
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord)
        if (etape%10==0): 
            name = f"{heure}_A_etape_{etape}_propagation_taille_{taille}.jpg"
            cv2.imshow(name,A)
            cv2.imwrite(dir_name+name,A)
            #visu=visualisation(A, FNN,taille,holes_coord, He)
            #cv2.imshow(f"Visu_propagation_{etape}",visu)
        
        print(f"Random search {etape} taille: {taille}")
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord)
        if (etape%10==0): 
            name = f"{heure}_A_etape_{etape}_random_taille_{taille}.jpg"
            cv2.imshow(name,A)
            print(dir_name+name)
            cv2.imwrite(dir_name+name,A)
            #visu=visualisation(A,FNN,taille,holes_coord, He)
            #cv2.imshow(f"Visu_random_{etape}",visu)
        print(f"Evaluation distance image origine etape {etape} : {evaluation(B_origine,A)}")
    print(evaluation(B_origine,A))
    cv2.waitKey()



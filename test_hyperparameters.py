import cv2
import os
import argparse
import numpy as np
import jupyterlab
import datetime
from time import strftime
from patchmatch import initialisation, propagation, random_search
from utils import gradient, visualisation, verification
from trackbar import get_hole, paint_draw, parameters
from matplotlib import pyplot as plt



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


def test_taille_patches(taille_patches, He, B, holes_coord,alpha):
    scale=15
    nombre_etape=9
    distances = {}
    #Nouveau dossier pour chaque etape
    for etape in range(nombre_etape+1):
        if (etape%3==0): 
            try:
                os.mkdir(f"./tests_hyperparameters/taille_patches/etape_{etape}")
            except OSError as error:
                pass          
    for taille in taille_patches:
        FNN, A = initialisation(He,B,taille,holes_coord)
        #PROPAGATION
        for etape in range(nombre_etape+1):
            FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
            FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
            if (etape%3==0): 
                cv2.imshow(f"Patch_{taille}_Random_nb_{etape}",A)
                cv2.imwrite( f"tests_hyperparameters/taille_patches/etape_{etape}/size_patch_{taille}.jpg", A )
            if(etape == nombre_etape):
                 distances[taille] = np.linalg.norm(B-A)
    return distances


def test_scales(scales, He, B,holes_coord,alpha):
    taille = 10
    nombre_etape=9
    distances = {}
    #Nouveau dossier pour chaque etape
    for etape in range(nombre_etape+1):
        if (etape%3==0): 
            try:
                os.mkdir(f"./tests_hyperparameters/scales/etape_{etape}")
            except OSError as error:
                pass
    for scale in scales:
        FNN, A = initialisation(He,B,taille,holes_coord)
        cv2.imshow(f"Test_Scale_{taille}",A)
        #PROPAGATION
        for etape in range(nombre_etape+1):
            FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
            FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
            if (etape%3==0): 
                cv2.imshow(f"Scale_{scale}_Random_nb_{etape}",A)
                cv2.imwrite( f"tests_hyperparameters/scales/etape_{etape}/size_scale_{scale}.jpg", A )
            if(etape == nombre_etape):
                distances[scale]=(np.linalg.norm(B-A))
    return distances


def test_nombre_etapes(nombre_etapes_max,He,B,holes_coord,alpha):
    taille = 10
    scale = 20
    distances = {}
    #INITIALISATION
    FNN, A = initialisation(He,B,taille,holes_coord)
    cv2.imshow(f"Tests_Nb_Etapes_{taille}",A)
    #PROPAGATION
    for etape in range(nombre_etapes_max+1):
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord, alpha)
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord,alpha)
        cv2.imshow(f"Random_nb_{etape}",A)
        cv2.imwrite( f"tests_hyperparameters/nombre_etapes/etape_{etape}.jpg", A)
        distances[etape]=np.linalg.norm(B-A)
    return distances



if __name__ == '__main__':
    #HYPERPARAMETERS
    taille_patches = [2,4,6,8,10,12,14,16,18]
    scales=[4,8,16,32,64,128,256,512]
    nombre_etapes_max=30

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
    heightB,widthB = B.shape[:2]
    cv2.rectangle(B,(taille,taille),(widthB-taille,heightB-taille),(233))
    cv2.imshow("image_B",B)
    grad = gradient(B)
    cv2.imshow("gradient",grad)

    #INITIALISATION DU TROU
    name = "Windows"
    cv2.imshow(name,B)
    param = get_hole(B,size_brush,name)
    holes_coord = {"x_min":param.x_min,"x_max":param.x_max,"y_min":param.y_min,"y_max":param.y_max}
    He =param.hole
    for x in range(param.width-1):
        for y in range(param.height-1):
            if He[y,x]==1:
                B[y,x]=(0,0,0)
    date = datetime.datetime.now()

    #CREATION DES DOSSIERS TESTS
    try:
        os.mkdir(f"./tests_hyperparameters")
        os.mkdir(f"./tests_hyperparameters/nombre_etapes")
        os.mkdir(f"./tests_hyperparameters/scales")
        os.mkdir(f"./tests_hyperparameters/taille_patches")
        os.mkdir(f"./tests_hyperparameters/taille_patches/figures")
        os.mkdir(f"./tests_hyperparameters/nombre_etapes/figures")
        os.mkdir(f"./tests_hyperparameters/scales/figures")
    except OSError as error:
                pass

    #TESTS
    distances_patches =test_taille_patches(taille_patches,He,B,holes_coord,alpha)
    distances_scales = test_scales(scales,He,B,holes_coord,alpha)
    distances_nb_etapes = test_nombre_etapes(nombre_etapes_max,He,B,holes_coord,alpha)
    distances = {"taille_patches":distances_patches, "scales":distances_scales, "nombre_etapes":distances_nb_etapes}
    
    #PLOT
    for key,tab in distances.items() :
        lists = sorted(tab.items()) # sorted by key, return a list of tuples
        metric, distance_to_B = zip(*lists)
        plt.figure()
        plt.plot(metric,distance_to_B)
        plt.ylabel('Distance_A_to_B')
        plt.xlabel(f'Metric : {key} ')
        plt.savefig(f'tests_hyperparameters/{key}/figures/test_{key}.jpg')
        count = count+1
    plt.show()
    cv2.waitKey()

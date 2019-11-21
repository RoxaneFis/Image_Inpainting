import cv2
import argparse
import numpy as np
import jupyterlab
from patchmatch import initialisation, propagation, random_search


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("Debut")
    taille=2
    scale=5
    nombre_etape=3
    args = parse_args()
    B =cv2.imread(args.image_input)
    He=B
    cv2.imshow("image_B",B)
    A = B
    FNN, A = initialisation(He,B,A,taille)
    cv2.imshow("image_A_ini",A)
    for etape in range(nombre_etape):
        print("Debut etape"+str(etape))
        print("Phase de propagation")
        FNN,A = propagation(B,A,FNN,etape,taille)
        cv2.imshow('image_A_propag '+str(etape),A)
        print("Phase de random search")
        FNN,A = random_search(B,A,FNN,taille,scale)
        cv2.imshow('image_A_random_search '+str(etape),A)
    cv2.waitKey()



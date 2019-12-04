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
    heightB,weightB = B.shape[:2]
    cv2.imshow("image_B",B)

    x_min=int(weightB/2)
    x_max=x_min + 30
    y_min=int(heightB/2)
    y_max=y_min + 30

    holes_coord = {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max}
    He = np.zeros((heightB,weightB),dtype='i')
    He[x_min:x_max,y_min:y_max]=1

    FNN, A_padding = initialisation(He,B,taille,holes_coord)
    cv2.imshow("image_A_padding_ini",A_padding)
    C = B
    C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
    cv2.imshow("image_C_ini",C)

    # for etape in range(nombre_etape):
    #     print("Debut etape"+str(etape))
    #     print("Phase de propagation")
    #     FNN,A_padding = propagation(B,A_padding,FNN,etape,taille)
    #     cv2.imshow('image_A_propag '+str(etape),A_padding)
    #     print("Phase de random search")
    #     FNN,A_padding = random_search(B,A_padding,FNN,taille,scale)
    #     cv2.imshow('image_A_random_search '+str(etape),A_padding)
    cv2.waitKey()



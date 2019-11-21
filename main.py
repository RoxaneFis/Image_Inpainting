import cv2
import argparse
import numpy as np
import jupyterlab
from patchmatch import initialisation, propagation


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("Debut")
    taille=20
    args = parse_args()
    B =cv2.imread(args.image_input)
    He=B
    cv2.imshow("image_B",B)
    A = B
    FNN, A = initialisation(He,B,A,taille)
    cv2.imshow("image_A_ini",A)
    FNN,A = propagation(B,A,FNN,0,taille)
    cv2.imshow('image_A_propag',A)
    FNN,A = propagation(B,A,FNN,1,taille)
    cv2.imshow('image_A_propag_2',A)
    cv2.waitKey()



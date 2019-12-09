import cv2
import argparse
import numpy as np
import jupyterlab
from patchmatch import initialisation, propagation, random_search
from utils import gradient, visualisation, verification


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    parser.add_argument('--nombre_etapes', type=int, help='nombre d etapes')
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("Debut")
    


    #INITIAL IMAGE
    args = parse_args()
    B =cv2.imread(args.image_input)
    heightB,widthB = B.shape[:2]
    cv2.imshow("image_B",B)
    grad = gradient(B)
    cv2.imshow("gradient",grad)

    #HYPERPARAMETERS
    taille=40
    scale=800
    nombre_etape=args.nombre_etapes

    #INITIALIZE HOLE
    x_min=int(widthB/2) 
    x_max=x_min + 80
    y_min=int(heightB/2)
    y_max=y_min + 80
    holes_coord = {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max}
    He = np.zeros((heightB,widthB),dtype='i')
    He[y_min:y_max+1,x_min:x_max+1]=1

    #INITIALISATION
    FNN, A = initialisation(He,B,taille,holes_coord)
    cv2.imshow("image_A_ini",A)

    #visu=visualisation(A_padding,C,FNN,taille,holes_coord)
    #cv2.imshow(f"Visu_propagation_ini",visu)

    for etape in range(nombre_etape):
        print(f"Phase de propagation: {etape}")
        FNN,A = propagation(A,He,FNN,etape,taille,holes_coord)
        
        if (etape%3==0): 
            cv2.imshow(f"A_propagation_{etape}",A)
            #visu=visualisation(A_padding,C,FNN,taille,holes_coord)
            #cv2.imshow(f"Visu_propagation_{etape}",visu)
        
        print(f"Random search {etape}")
        FNN,A = random_search(A, He,FNN,taille,scale,holes_coord)
        if (etape%3==0): 
            cv2.imshow(f"A_random_{etape}",A)
            #visu=visualisation(A_padding,C,FNN,taille,holes_coord)
            #cv2.imshow(f"Visu_random_{etape}",visu)
        #verification(A_padding,C,FNN,taille,holes_coord)
    cv2.waitKey()



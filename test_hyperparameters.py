import os
import cv2
import argparse
import numpy as np
import jupyterlab
import matplotlib.pyplot as plt
from patchmatch import initialisation, propagation, random_search


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    parser.add_argument('--exp_nb', type=int, help='file to full image to fill')
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

def test_taille_patches(taille_patches, He, B):
    scale=15
    nombre_etape=9
    C = B
    B_copy = B.copy()
    distances = {}
    print(f"\nTEST_TAILLE_PATCHES\nHyperparametres:\nNombre_etape={nombre_etape}\nScale={scale}\n")
    #Nouveau dossier pour chaque etape
    for etape in range(nombre_etape+1):
        if (etape%3==0): 
            try:
                os.mkdir(f"./tests_hyperparameters/taille_patches/etape_{etape}")
            except OSError as error:
                pass
    for taille in taille_patches:
        #print(f"Test taille patch : {taille}")
        #INITIALISATION
        FNN, A_padding = initialisation(He,B,taille,holes_coord)
        C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
        cv2.imshow(f"Test_taille_patch_{taille}",C)
        
        #PROPAGATION
        for etape in range(nombre_etape+1):
            FNN,A_padding = propagation(B,A_padding,He,FNN,etape,taille)
            C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
            if (etape%3==0): 
                pass
                cv2.imshow(f"Patch_{taille}_Propagation_nb_{etape}",C)
            FNN,A_padding = random_search(B,A_padding, He,FNN,taille,scale)
            C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
            if (etape%3==0): 
                cv2.imshow(f"Patch_{taille}_Random_nb_{etape}",C)
                cv2.imwrite( f"tests_hyperparameters/taille_patches/etape_{etape}/size_patch_{taille}.jpg", C )
            if(etape == nombre_etape):
                 distances[taille] = np.linalg.norm(B_copy-C)
    print("Fin : Les images ont bien été sauvegardé dans tests_hyperparameters/taille_patches ")
    return distances


def test_scales(scales, He, B):
    taille = 8
    nombre_etape=9
    distances = {}
    B_copy = B.copy()
    print(f"\nTEST_SCALE\nHyperparametres:\nTaille_patch={taille}\nNombre_etape={nombre_etape}\n")
    #Nouveau dossier pour chaque etape
    for etape in range(nombre_etape+1):
        if (etape%3==0): 
            try:
                os.mkdir(f"./tests_hyperparameters/scales/etape_{etape}")
            except OSError as error:
                pass

    for scale in scales:
        #print(f"Test scale : {scale}")
        #INITIALISATION
        FNN, A_padding = initialisation(He,B,taille,holes_coord)
        C = B
        C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
        cv2.imshow(f"Test_Scale_{taille}",C)
        #PROPAGATION
        for etape in range(nombre_etape+1):
            FNN,A_padding = propagation(B,A_padding,He,FNN,etape,taille)
            if (etape%3==0): 
                C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
                cv2.imshow(f"Scale_{scale}_Propagation_nb_{etape}",C)
            FNN,A_padding = random_search(B,A_padding, He,FNN,taille,scale)
            if (etape%3==0): 
                C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
                cv2.imshow(f"Scale_{scale}_Random_nb_{etape}",C)
                cv2.imwrite( f"tests_hyperparameters/scales/etape_{etape}/size_scale_{scale}.jpg", C )
            if(etape == nombre_etape):
                distances[scale]=(np.linalg.norm(B_copy-C))
    print("Fin : Les images ont bien été sauvegardé dans tests_hyperparameters/scales ")
    return distances


def test_nombre_etapes(nombre_etapes_max,He,B):
    taille = 8
    scale = 20
    distances = {}
    print(f"\nTEST_NOMBRES_ETAPES\nHyperparametres:\nTaille_patch={taille}\nScale={scale}\n")
    #INITIALISATION
    FNN, A_padding = initialisation(He,B,taille,holes_coord)
    C = B
    B_copy = B.copy()
    C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
    cv2.imshow(f"Tests_Nb_Etapes_{taille}",C)
    #PROPAGATION
    for etape in range(nombre_etapes_max+1):
        #Propa
        FNN,A_padding = propagation(B,A_padding,He,FNN,etape,taille)
        C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
        cv2.imshow(f"Propagation_nb_{etape}",C)
        #Random_search
        FNN,A_padding = random_search(B,A_padding, He,FNN,taille,scale)
        C[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]=A_padding
        cv2.imshow(f"Random_nb_{etape}",C)
        cv2.imwrite( f"tests_hyperparameters/nombre_etapes/etape_{etape}.jpg", C )
        distances[etape]=np.linalg.norm(B_copy-C)
    print("Fin : Les images ont bien été sauvegardé dans tests_hyperparameters/nb_etapes ")
    return distances



if __name__ == '__main__':
    #HYPERPARAMETERS
    taille_patches = [2,4,6,8,10,12]
    scales=[4,8,12,16,20,24,26,28]
    nombre_etapes_max=30

    #INITIAL IMAGE
    args = parse_args()
    B =cv2.imread(args.image_input)
    heightB,weightB = B.shape[:2]
    cv2.imshow("image_B",B)

    #INITIALIZE HOLE
    x_min=int(weightB/2)
    x_max=x_min + 30
    y_min=int(heightB/2)
    y_max=y_min + 30
    holes_coord = {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max}
    He = np.zeros((heightB,weightB),dtype='i')
    He[x_min:x_max,y_min:y_max]=1

    #TESTS
    distances_patches =test_taille_patches(taille_patches,He,B)
    distances_scales = test_scales(scales,He,B)
    distances_nb_etapes = test_nombre_etapes(nombre_etapes_max,He,B)
    distances = {"taille_patches":distances_patches, "scales":distances_scales, "nombre_etapes":distances_nb_etapes}
    
    #PLOT
    for key,tab in distances.items() :
        lists = sorted(tab.items()) # sorted by key, return a list of tuples
        metric, distance_to_B = zip(*lists)
        plt.figure()
        plt.plot(metric,distance_to_B)
        plt.ylabel('Distance_A_to_B')
        plt.xlabel(f'Metric : {key} ')
        plt.savefig(f'tests_hyperparameters/{key}/figures/test_nb_{args.exp_nb}.jpg')
    plt.show()

    cv2.waitKey()

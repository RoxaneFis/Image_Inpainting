import numpy as np
import cv2

class Patch:
    def __init__(self,taille,matrice,y,x):
        self.taille = taille
        self.mat = matrice[y-taille:y+taille+1,x-taille:x+taille+1]
    def distance(self, P2):
        d = cv2.norm(self.mat,P2.mat)
        return d
        
import numpy as np
import cv2
from utils import gradient

class Patch:
    def __init__(self,taille,matrice,y,x):
        self.taille = taille
        self.mat = matrice[y-taille:y+taille+1,x-taille:x+taille+1]
        self.grad = gradient(self.mat)
    def distance(self, P2,alpha):
        #alpha : entre 0 et 1, quantifie l'importance relative du gradient. alpha=1 : ne prend pas en compte le gradient
        d_color = cv2.norm(self.mat,P2.mat)
        d_gradient =  cv2.norm(self.grad,P2.grad)
        d = alpha*d_color+(1-alpha)*d_gradient
        return d
        
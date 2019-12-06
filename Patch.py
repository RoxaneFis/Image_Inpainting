import numpy as np
import cv2
from utils import gradient
alpha=[0.7,0.3]
class Patch:
    def __init__(self,taille,matrice,y,x):
        self.taille = taille
        self.mat = matrice[y-taille:y+taille+1,x-taille:x+taille+1]
        #self.grad = gradient(self.mat)
    def distance(self, P2):
        d_color = cv2.norm(self.mat,P2.mat)
        #d_gradient =  cv2.norm(self.grad,P2.grad)
        #d = alpha[0]*d_color+alpha[1]*d_gradient
        return d_color
        
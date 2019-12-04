import random
import numpy as np
import cv2
from Patch import Patch

def initialisation(He, B, A,taille):
    #B image trouee
    #A image à randomiser
    heightA,weightA = A.shape[:2]
    heightB,weightB = B.shape[:2]
    FNN = np.zeros((heightB,weightB,2),dtype='i')
    for x in range(weightA):
        for y in range(heightA):
            xB = random.randint(taille,weightB-1-taille)
            yB = random.randint(taille,heightB-1-taille)
            FNN[y][x][0]=yB
            FNN[y][x][1]=xB
            A[y,x]=B[yB,xB]
    return FNN,A

def propagation(B,A,FNN,etape,taille):
    heightA,weightA = A.shape[:2]
    heightB,weightB = B.shape[:2]
    if (etape%2==0): #pair
        for x in range(taille+1,weightA-taille-1):
            for y in range(taille+1,heightA-taille-1):
                    PA = Patch(taille,A,y,x)
                    voisins=[[y,x],[y-1,x],[y,x-1]]
                    #TO DO : essayer avec les voisins de FNN[y-1,x] et FNN[y,x-1] (attention aux bords)
                    voisins_B = [FNN[y,x],FNN[y-1,x]+[1,0],FNN[y,x-1]+[0,1]]
                    if(voisins_B[1][0]>heightB-taille-1):
                        voisins_B[1]=FNN[y-1,x]
                    if(voisins_B[2][1]>weightB-taille-1):
                        voisins_B[2]=FNN[y,x-1]
                    distances=[]
                    for i in range(3):
                        PB = Patch(taille,B,voisins_B[i][0],voisins_B[i][1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_B[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A[y,x]=B[yB,xB]

    if (etape%2==1): #impair
        for y in range(heightA-taille-2,taille):
            for x in range(weightA-taille-2,taille):
                    PA = Patch(taille,A,y,x)
                    voisins=[[y,x],[y+1,x],[y,x+1]]
                    voisins_B = [FNN[y,x],FNN[y+1,x],FNN[y,x+1]]
                    distances=[]
                    for i in range(3):
                        PB = Patch(taille,B,voisins_B[i][0],voisins_B[i][1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_B[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A[y,x]=B[yB,xB]
    return FNN,A

def random_search(B,A,FNN,taille,scale):
    #Phase à modifier : modification taille de la fenetre à diminuer au sein d'une même étape + RANDOM
    heightA,weightA = A.shape[:2]
    heightB,weightB = B.shape[:2]
    for x in range(taille+scale+1,weightA-scale-taille-2):
        for y in range(taille+scale+1,heightA-scale-taille-2):
            PA = Patch(taille,A,y,x)
            scale_reduce = scale
            while (scale_reduce >=1):
                xB = FNN[y,x][1]
                yB = FNN[y,x][0]
                rx = random.randint(-scale,scale)
                ry = random.randint(-scale,scale)
                #TO Do : vérifier que rx et ry ne sont pas dans le trou ou en dehors du bord

                PB = Patch(taille,B,yB,xB)
                P_Potentiel = Patch(taille,B,yB+ry,xB+rx)
                if PA.distance(P_Potentiel)<PA.distance(PB):
                    FNN[y][x][0]=yB+ry
                    FNN[y][x][1]=xB+rx
                    A[y,x]=B[yB+ry,xB+rx]

                scale_reduce=scale_reduce//2
    return FNN,A

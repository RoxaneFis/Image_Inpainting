import random
import numpy as np
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
                    voisins_B = [FNN[y,x],FNN[y-1,x],FNN[y,x-1]]
                    distances=[]
                    for i in range(3):
                        PB = Patch(taille,B,voisins_B[i][0],voisins_B[i][1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_B[ind_min]
                    if(ind_min==1):
                        A[y,x]=B[y-1,x]
                    if(ind_min==2):
                        A[y,x]=B[y,x-1]

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
                    if(ind_min==1):
                        A[y,x]=B[y+1,x]
                        print("chgt")
                    if(ind_min==2):
                        print("chgt")
                        A[y,x]=B[y,x+1]
    return FNN,A

def random_search(B,A,FNN,taille,scale):
    heightA,weightA = A.shape[:2]
    heightB,weightB = B.shape[:2]
    for x in range(taille+scale+1,weightA-scale-taille-2):
        for y in range(taille+scale+1,heightA-scale-taille-2):
            min_dist=float("inf")
            PA = Patch(taille,A,y,x)
            ind_i=-scale
            ind_j=-scale
            for i in range(-scale,scale+1):
                for j in range(-scale,scale+1):
                    PB = Patch(taille,B,y+j,x+i)
                    d = PA.distance(PB)
                    if(d<min_dist):
                        min_dist=d
                        ind_i=i
                        ind_j=j
            FNN[y,x]=FNN[y+ind_j,x+ind_i]
            A[y,x]=B[y+ind_j,x+ind_i]
    return FNN,A

import random
import numpy as np
import cv2
from Patch import Patch
from utils import gradient

def check_boundary(taille:int,y:int, x:int,height:int,width:int)->bool:
    return(y>=taille and y<height-taille and x>=taille and x<width-taille)

def not_hole(taille:int,He, y:int, x:int)->bool:
    height,width = He.shape[:2]
    if(not check_boundary(taille,y,x,height,width)or He[y,x]==1 ):
        return False
    else:
        return True

        

def initialisation(He, B, taille: int, holes_coord):
    #B image trouee
    #A image à randomiser
    x_min = holes_coord["x_min"]
    x_max = holes_coord["x_max"]
    y_min = holes_coord["y_min"]
    y_max = holes_coord["y_max"]
    A = B.copy()
    heightB,widthB = B.shape[:2]
    FNN = np.zeros((heightB,widthB,2),dtype='i')
    for x in range(widthB):
        for y in range(heightB):
            FNN[y][x][0]=y
            FNN[y][x][1]=x
    #Initilisation de A : pixel aleatoire dans les trous

    for x in range(x_min,x_max):
        for y in range(y_min,y_max):
            if He[y][x]==1:
                xB = random.randint(taille,widthB-1-taille)
                yB = random.randint(taille,heightB-1-taille)
                while (not not_hole(taille,He,yB,xB)):
                    xB = random.randint(taille,widthB-1-taille)
                    yB = random.randint(taille,heightB-1-taille)
                #Initialise Fnn with random patch values (not in hole)
                FNN[y][x][0]=yB
                FNN[y][x][1]=xB
                #Initialise A with random values
                A[y,x]=B[yB,xB]
    return FNN,A

def propagation(A,He,FNN,etape,taille,holes_coord):
    x_min = holes_coord["x_min"]
    x_max = holes_coord["x_max"]
    y_min = holes_coord["y_min"]
    y_max = holes_coord["y_max"]
    heightA,widthA = A.shape[:2]
    if (etape%2==0): #pair
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if He[y][x]==1:
                    PA = Patch(taille,A,y,x)
                    voisins=[[y,x],[y-1,x],[y,x-1]]
                    #TO DO : essayer avec les voisins de FNN[y-1,x] et FNN[y,x-1] (attention aux bords)
                    voisins_FNN = [FNN[y,x],FNN[y-1,x],FNN[y-1,x]+[1,0],FNN[y,x-1],FNN[y,x-1]+[0,1]]
                    #voisins_FNN = [FNN[y,x],FNN[y-1,x],FNN[y,x-1]]
                    voisins_FNN = [indices for indices in voisins_FNN if not_hole(taille,He, indices[0], indices[1]) ]
                    distances=[]
                    for voisin in voisins_FNN:
                        PB = Patch(taille,A,voisin[0],voisin[1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    #print(f"indice_min : {ind_min}")
                    #if(voisins_FNN[ind_min][0]==(FNN[y-1,x]+[1,0])[0] and voisins_FNN[ind_min][1]==(FNN[y-1,x]+[1,0])[1]):
                    #    print(f"Amelioration du voisin du patch du voisin pour le pixel : x={x},y={y}")
                    #if(voisins_FNN[ind_min][0]==(FNN[y,x-1]+[0,1])[0] and voisins_FNN[ind_min][1]==(FNN[y,x-1]+[0,1])[1]):
                    #    print(f"Amelioration du voisin du patch du voisin pour le pixel : x={x},y={y}")
                    FNN[y,x]=voisins_FNN[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A[y,x]=A[yB,xB]

    if (etape%2==1): #impair
        for x in range(x_max,x_min,-1):
            for y in range(y_max,y_min,-1):
                if He[y][x]==1:
                    PA = Patch(taille,A,y,x)
                    voisins=[[y,x],[y+1,x],[y,x+1]]
                    voisins_FNN = [FNN[y,x],FNN[y+1,x],FNN[y+1,x]+[-1,0],FNN[y,x+1],FNN[y,x+1]+[0,-1]]
                    #voisins_FNN = [FNN[y,x],FNN[y+1,x],FNN[y,x+1]]
                    voisins_FNN = [indices for indices in voisins_FNN if not_hole(taille,He, indices[0], indices[1]) ]                    
                    distances=[]
                    for voisin in voisins_FNN:
                        PB = Patch(taille,A,voisin[0],voisin[1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_FNN[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A[y,x]=A[yB,xB]
    return FNN,A

def random_search(A,He,FNN,taille,scale,holes_coord):
    x_min = holes_coord["x_min"]
    x_max = holes_coord["x_max"]
    y_min = holes_coord["y_min"]
    y_max = holes_coord["y_max"]
    heightA,widthA = A.shape[:2]
    for x in range(x_min,x_max):
        for y in range(y_min,y_max):
            if He[y][x]==1:
                PA = Patch(taille,A,y,x)
                scale_reduce = scale
                while (scale_reduce >=1):
                    xB = FNN[y,x][1]
                    yB = FNN[y,x][0]
                    rx = random.randint(-scale,scale)
                    ry = random.randint(-scale,scale)
                    while (not not_hole(taille,He,yB+ry,xB+rx)):
                        rx = random.randint(-scale,scale)
                        ry = random.randint(-scale,scale)
                    PB = Patch(taille,A,yB,xB)
                    P_Potentiel = Patch(taille,A,yB+ry,xB+rx)
                    if PA.distance(P_Potentiel)<PA.distance(PB):
                        FNN[y][x][0]=yB+ry
                        FNN[y][x][1]=xB+rx
                        A[y,x]=A[yB+ry,xB+rx]
                    scale_reduce=(3*scale_reduce)//4
    return FNN,A

import random
import numpy as np
import cv2
from Patch import Patch

def check_boundary(x:int, y:int,height:int,width:int)->bool:
    return(x>=0 and x<height and y>=0 and y<width)

def not_hole(He, x:int, y:int)->bool:
    height,width = He.shape[:2]
    if(He[y,x]==1 or not check_boundary(x,y,height,width)):
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
    heightA = x_max- x_min+ 1
    widthA = y_max-y_min+1
    heightB,widthB = B.shape[:2]
    FNN = np.zeros((heightA,widthA,2),dtype='i')
    #Initialise A_padding = holes + boundaries
    A_padding = np.zeros((heightA+2*taille,widthA+2*taille))
    A_padding = B[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]
    for x in range(taille, widthA+taille):
        for y in range(taille, heightA+taille):
            xB = random.randint(taille,widthB-1-taille)
            yB = random.randint(taille,heightB-1-taille)
            while (not not_hole(He,xB,yB)):
                xB = random.randint(taille,widthB-1-taille)
                yB = random.randint(taille,heightB-1-taille)
            #Initialise Fnn with random patch values
            FNN[y-taille][x-taille][0]=yB
            FNN[y-taille][x-taille][1]=xB
            #Initialise A with random values
            A_padding[y,x]=B[yB,xB]
    return FNN,A_padding

def propagation(B,A,FNN,etape,taille):
    heightA,widthA = A.shape[:2]
    heightB,widthB = B.shape[:2]
    if (etape%2==0): #pair
        for x in range(taille+1,widthA-taille-1):
            for y in range(taille+1,heightA-taille-1):
                    PA = Patch(taille,A,y,x)
                    voisins=[[y,x],[y-1,x],[y,x-1]]
                    #TO DO : essayer avec les voisins de FNN[y-1,x] et FNN[y,x-1] (attention aux bords)
                    voisins_B = [FNN[y,x],FNN[y-1,x]+[1,0],FNN[y,x-1]+[0,1]]
                    if(voisins_B[1][0]>heightB-taille-1):
                        voisins_B[1]=FNN[y-1,x]
                    if(voisins_B[2][1]>widthB-taille-1):
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
            for x in range(widthA-taille-2,taille):
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
    heightA,widthA = A.shape[:2]
    heightB,widthB = B.shape[:2]

    for x in range(taille+scale+1,widthA-scale-taille-2):
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
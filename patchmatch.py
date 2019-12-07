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
    heightA = y_max- y_min+ 1
    widthA = x_max-x_min+1
    heightB,widthB = B.shape[:2]
    FNN = np.zeros((heightA+2*taille,widthA+2*taille,2),dtype='i')
    for x in range(widthA+2*taille):
        for y in range(heightA+2*taille):
            FNN[y][x][0]=y+y_min-taille
            FNN[y][x][1]=x+x_min-taille

    #Initialise A_padding = holes + boundaries
    A_padding = np.zeros((heightA+2*taille,widthA+2*taille))
    A_padding = B[y_min-taille:y_max+taille+1,x_min-taille:x_max+taille+1]
    
    for x in range(taille, widthA+taille):
        for y in range(taille, heightA+taille):
            xB = random.randint(taille,widthB-1-taille)
            yB = random.randint(taille,heightB-1-taille)
            while (not not_hole(taille,He,yB,xB)):
                xB = random.randint(taille,widthB-1-taille)
                yB = random.randint(taille,heightB-1-taille)
            #Initialise Fnn with random patch values (not in hole)
            FNN[y][x][0]=yB
            FNN[y][x][1]=xB
            #Initialise A with random values
            A_padding[y,x]=B[yB,xB]
    return FNN,A_padding

def propagation(B,A_padding,He,FNN,etape,taille):
    heightA_padding,widthA_padding = A_padding.shape[:2]
    heightA,widthA = heightA_padding-2*taille,widthA_padding-2*taille
    heightB,widthB = B.shape[:2]
    if (etape%2==0): #pair
        for x in range(taille, widthA+taille):
            print(x)
            for y in range(taille, heightA+taille):
                    PA = Patch(taille,A_padding,y,x)
                    voisins=[[y,x],[y-1,x],[y,x-1]]
                    #TO DO : essayer avec les voisins de FNN[y-1,x] et FNN[y,x-1] (attention aux bords)
                    voisins_B = [FNN[y,x],FNN[y-1,x],FNN[y,x-1]]
                    voisins_B = [indices for indices in voisins_B if not_hole(taille,He, indices[0], indices[1]) ]
                    distances=[]
                    for voisin in voisins_B:
                        PB = Patch(taille,B,voisin[0],voisin[1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_B[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A_padding[y,x]=B[yB,xB]

    if (etape%2==1): #impair
        for x in range(widthA+taille-1,taille-1,-1):
            print(x)
            for y in range(heightA+taille-1,taille-1,-1):
                    PA = Patch(taille,A_padding,y,x)
                    voisins=[[y,x],[y+1,x],[y,x+1]]
                    voisins_B = [FNN[y,x],FNN[y+1,x],FNN[y,x+1]]
                    voisins_B = [indices for indices in voisins_B if not_hole(taille,He, indices[0], indices[1]) ]                    
                    distances=[]
                    for voisin in voisins_B:
                        PB = Patch(taille,B,voisin[0],voisin[1])
                        distances.append(PA.distance(PB))
                    ind_min = np.argmin(distances)
                    FNN[y,x]=voisins_B[ind_min]
                    yB = FNN[y][x][0]
                    xB = FNN[y][x][1]
                    A_padding[y,x]=B[yB,xB]
    return FNN,A_padding

def random_search(B,A_padding,He,FNN,taille,scale):
    #Phase à modifier : modification taille de la fenetre à diminuer au sein d'une même étape + RANDOM
    heightA_padding,widthA_padding = A_padding.shape[:2]
    heightA,widthA = heightA_padding-2*taille,widthA_padding-2*taille
    heightB,widthB = B.shape[:2]
    i=0
    for x in range(taille, widthA+taille):
        for y in range(taille, heightA+taille):
            PA = Patch(taille,A_padding,y,x)
            scale_reduce = scale
            while (scale_reduce >=1):
                xB = FNN[y,x][1]
                yB = FNN[y,x][0]
                rx = random.randint(-scale,scale)
                ry = random.randint(-scale,scale)
                #TO Do : vérifier que rx et ry ne sont pas dans le trou ou en dehors du bord
                while (not not_hole(taille,He,yB+ry,xB+rx)):
                    rx = random.randint(-scale,scale)
                    ry = random.randint(-scale,scale)
                PB = Patch(taille,B,yB,xB)
                P_Potentiel = Patch(taille,B,yB+ry,xB+rx)
                # print("PB_______")
                # print(PB.mat)
                # print("P_Potentiel_______")
                # print(P_Potentiel.mat)
                # print(f"yb {yB}, xB {xB} ,yB+ry {yB+ry},xB+rx {xB+rx}")
                # print()
                
                if PA.distance(P_Potentiel)<PA.distance(PB):
                    print("AMELIORATION PAR RANDOM SEARCH pixel "+str(x) + ' '+str(y))
                    FNN[y][x][0]=yB+ry
                    FNN[y][x][1]=xB+rx
                    A_padding[y,x]=B[yB+ry,xB+rx]
                    i+=1

                scale_reduce=(3*scale_reduce)//4
    return FNN,A_padding

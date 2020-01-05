# Image Inpainting
Algorithme d'autocomplétion d'image s'appuyant sur l'algorithme PatchMatch


## Structure
Le projet contient les fichiers suivants :
* main.py : lancer l'algorithme, 
* Patch.py : classe des patchs
* trackbar.py : interface avec l'utilisateur pour tracer les trous
* utils.py : fonctions de visualisation, vérification, évaluation
* test_....py : tester les améliorations implémentées et les hyperparamètres


## Lancement du programme
Pour lancer le programme, il faut écrire la ligne de commande suivante :

    python main.py --image_input IMAGE.JPG --dir_name REPTERTOIRE_POUR_ENREGISTRER_LES_IMAGES --nombre_etapes 10 --taille 12 --sizebrush 10

Une fenêtre nommée "Windows" s'ouvre alors avec l'image, et l'utilisateur peut tracer les trous qu'il souhaite appliquer à l'image.


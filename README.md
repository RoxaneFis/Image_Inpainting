# Image Inpainting
Inpainting is the automatic reconstruction of missing or damaged parts of an image.

 ![](results/example1.png)

## Structure
The project contains the following files:
* main.py: launch the algorithm, 
* Patch.py: class of patches
* trackbar.py: user interface for tracing holes
* utils.py: visualization, verification, evaluation functions
* test.py: test the implemented improvements and hyperparameters

 ![](code_structure.png)


## Launching the programme
To start the program, write the following command line :
 ```js
python main.py --image_input <IMAGE.JPG> --dir_name <dir_outputs> --nombre_etapes 10 --taille 12 --sizebrush 10
```

A window named "windows" then opens with the image, and the user can draw the holes he wants to apply to the image.

 ![](results/step_ini.jpg)


Progressive steps:

Step 1         |  Step 2 
:-------------------------:|:-------------------------: 
 ![](results/step_0.jpg)  |   ![](results/step_1.jpg) |  

Step 3         |  Step 4
:-------------------------:|:-------------------------: 
 ![](results/step_2.jpg)  |   ![](results/step_3.jpg) |  

 Step 8        |  Step 9
:-------------------------:|:-------------------------: 
 ![](results/step_8.jpg)  |   ![](results/step_9.jpg) |  



import cv2
import numpy as np

drawing=False # true if mouse is pressed
mode=False

class parameters:
    def __init__(self, _image,_size_brush):
        self.image=_image
        self.image_copy = _image.copy()
        height,width = _image.shape[:2]
        self.height = height
        self.width = width
        self.hole = np.zeros((height,width),dtype='i')
        self.size_brush = _size_brush
        self.x_min=width
        self.x_max=0
        self.y_min=height
        self.y_max=0


# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        mode = True
        current_former_x,current_former_y=former_x,former_y
        cv2.line(param.image_copy,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),param.size_brush)
        param.hole[former_y:former_y+param.size_brush,former_x:former_x+param.size_brush]=1
                
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                if(param.x_min>former_x-param.size_brush):
                    param.x_min = former_x-param.size_brush
                if(param.x_max<former_x+param.size_brush):
                    param.x_max = former_x+param.size_brush
                if(param.y_min>former_y-param.size_brush):
                    param.y_min = former_y-param.size_brush
                if(param.y_max<former_y+param.size_brush):
                    param.y_max = former_y+param.size_brush
                cv2.line(param.image_copy,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),param.size_brush)
                current_former_x = former_x
                current_former_y = former_y
                param.hole[former_y:former_y+param.size_brush,former_x:former_x+param.size_brush]=1
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            current_former_x = former_x
            current_former_y = former_y


    return former_x,former_y


def get_hole(image,size_brush,name):
    param = parameters(image,size_brush)
    cv2.setMouseCallback(name,paint_draw,param)
    
    while(1):
        cv2.imshow(name,param.image_copy)

        hole = 100*np.stack((param.hole,) * 3,-1)
        hole = hole.astype(np.uint8)
        gray = cv2.cvtColor(hole, cv2.COLOR_BGR2GRAY)

        cv2.imshow('OpenCV Paint hole',gray)
        k=cv2.waitKey(10)& 0xFF
        if k == ord('r'):
            # Press key `q` to quit the program
            return param
            exit() 
    return param



if __name__ == "__main__":
    size_brush =10
    image = cv2.imread("s.jpg")
    cv2.imshow('OpenCV Paint Brush',image)
    param = get_hole(image,size_brush)
    









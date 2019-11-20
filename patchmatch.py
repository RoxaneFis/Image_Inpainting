import cv2
import argparse
import numpy as np


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    #parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    image =cv2.imread(args.image_input)
    cv2.imshow("image",image)
    cv2.waitKey()



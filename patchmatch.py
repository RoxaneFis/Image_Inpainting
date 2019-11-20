import cv2
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--image_input', type=str, help='file to full image to fill')
    parser.add_argument('--image_hole', type=str, help='file to hole in full image to fill')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    image = cv2.imread(args.input_image)
    hole = cv2.imread(args.image_hole)
    image.imshow()


    #Initialisation : chaque pixel
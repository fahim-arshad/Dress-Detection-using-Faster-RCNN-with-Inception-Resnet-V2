import numpy as np
import cv2
import os

dir_path = os.getcwd()

for filename in os.listdir(dir_path):
   
    if filename.endswith(".JPG"):
        image = cv2.imread(filename)
        resized = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename,resized)

import cv2.cv2 as cv
import os
from time import sleep

camera=cv.VideoCapture(0)
_,fram=camera.read()
while 1:
    cv.imshow("camera",fram)
    cv.waitKey()
    cv.destroyAllWindows()
    


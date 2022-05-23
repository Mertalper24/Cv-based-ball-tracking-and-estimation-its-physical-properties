import cv2
import numpy as np
import cvzone
import math
from cvzone.ColorModule import ColorFinder

video = cv2.VideoCapture('Videos/video1.mp4')
Masking = ColorFinder(True)
hsvValues= {'hmin': 0, 'smin': 101, 'vmin': 198, 'hmax': 21, 'smax': 255, 'vmax': 255}

while True:
    success, ball_video = video.read()


    #Finding the hsvValues and initial masking via the image

    img=cv2.imread("ballimage.jpeg")
    colors , mask_initial = Masking.update(img,hsvValues)
    img=cv2.resize(img, (0,0),None, 0.6, 0.6)
    colors=cv2.resize(colors, (0,0),None, 0.6, 0.6)
    #cv2.imshow("image", img)
    cv2.imshow("colors",colors)
    cv2.waitKey(75)
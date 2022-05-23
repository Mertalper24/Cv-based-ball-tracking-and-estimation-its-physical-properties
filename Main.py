import cv2
import numpy as np
import cvzone
import math
from cvzone.ColorModule import ColorFinder

i = list(range(1,7))
for n in i:
    video = cv2.VideoCapture('Videos/video' + str(n) + '.mp4')
    Masking = ColorFinder(False)
    hsvValues= {'hmin': 0, 'smin': 101, 'vmin': 198, 'hmax': 21, 'smax': 255, 'vmax': 255}

    position_all = []
    posx = []
    posy = []
    time=[]
    x_frame = list(range(0,1920))
    while True:
        success, ball_video = video.read()

        #Applying masking to the video
        track_ball, mask = Masking.update(ball_video, hsvValues)

        #Finding the ball
        display, contours = cvzone.findContours(ball_video,mask, minArea=100)
        
        if contours:
            if posy:
                if posy[-1] > 750:
                    position_all.clear()
                    posx.clear()
                    posy.clear()

            position = contours[0]['center']
            if n==4:
                if len(time) == 33:
                    break
            elif len(time) == 26:
                    break
            position_all.append(position)
            posx.append(position[0])
            posy.append(position[1])
            time.append(position[0])


        # Quadratic equation of the movement of the ball

        if position_all:
            A,B,C = np.polyfit(posx,posy,2)

            for i, pos in enumerate (position_all):
                if (i>0):
                    cv2.circle(display,pos,5,(0,255,0),cv2.FILLED)
                    cv2.line(display, pos, position_all[i-1],(0,255,0),2)

            for x in x_frame:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(display, (x, y), 2, (255, 0, 255), cv2.FILLED)

            #Score

            C = C - 685
        
            x = int((-B - math.sqrt(B ** 2 - (4 * A * C))) / (2 * A))
            prediction = 470 < x < 600
        

            if position[1] < 750:
                if prediction:
                        cvzone.putTextRect(display, "Goes in", (50, 150),
                                        scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                else:
                        cvzone.putTextRect(display, "OUT", (50, 150),
                                        scale=5, thickness=5, colorR=(0, 0, 200), offset=20)
        #Display
        display = cv2.resize(display, (0,0),None, 0.6, 0.6)
        cv2.imshow("Ball", display)
        cv2.waitKey(75)
    video.release()
    cv2.destroyAllWindows()
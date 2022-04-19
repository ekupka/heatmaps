import cv2
import numpy as np
import time
import random
from collections import deque

cap = cv2.VideoCapture("fulll.mp4")

maskImage = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
bgmask = cv2.threshold(maskImage, 120, 255, cv2.THRESH_BINARY)[1]

ret0, frame0 = cap.read()
frames_queue = deque([frame0], maxlen=40)


background = frame0


def build_background():
    median= np.median(frames_queue, axis=0).astype(dtype=np.uint8)
    return median

def blur_image(img, num):
    return cv2.GaussianBlur(img, (num, num), 5, None, 3, 2 )

kernelX = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))


accumulated = np.zeros_like(maskImage, dtype=np.uint8)
accumulatorLength = 0
newAccumulated = np.zeros_like(maskImage, dtype=np.uint16)


accumulated = np.zeros_like(maskImage, dtype=np.int64)


while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    frames_queue.append(frame)


    #background = build_background()

    difference = cv2.absdiff(blur_image(background,3), blur_image(frame,3))
 
    background = frame

    g_dif = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
 
    thresh = cv2.threshold(g_dif, 20, 255, cv2.THRESH_BINARY)[1]
    
    frame_heat = cv2.subtract(thresh, bgmask)

    dilated = cv2.dilate(frame_heat, kernelX, iterations=1)
    
    newMask = blur_image(dilated, 7)

    cv2.imshow("thresh", thresh)
    cv2.imshow("thresh", newMask)


    #weight1 = accumulatorLength / (accumulatorLength + 1)
    #weight2 = 1 / (accumulatorLength + 1)
    #if (accumulatorLength==0): accumulated = newMask
    #else: accumulated = cv2.addWeighted(accumulated, weight1, newMask, weight2, 0)
    #accumulatorLength+=1
    
    accumulated = accumulated + newMask

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


weighted = accumulated * (256/accumulated.max())

map = weighted.astype(np.uint8)


heatmap = cv2.applyColorMap(map, cv2.COLORMAP_JET)

cv2.imshow('Result', heatmap)
cv2.imwrite("result3.png", heatmap)



print("OK")
cv2.waitKey(0)
cv2.destroyAllWindows()

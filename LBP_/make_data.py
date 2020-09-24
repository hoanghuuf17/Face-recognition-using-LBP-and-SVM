import numpy as np
import cv2
import time

label = "humman2"

cap = cv2.VideoCapture(0)

i=0
while(True):
    i+=1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None,fx=1,fy=1)

    cv2.imshow('frame',frame)

    if i>=1:
        print("so anh capture = ",i)
        cv2.imwrite('Image/' + str(label) + "/" + str(i) + ".jpg",frame)

    if i==100:
        break

cap.release()
cv2.destroyAllWindows()
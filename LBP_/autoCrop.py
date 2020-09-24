import cv2
import numpy as np 
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('img/16.jpg')

# image = Image.fromarray(img, "RGB")
# print(type(image))

image_copy = np.copy(img)
gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(img, 1.1, 4)
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop.append(gray_image[y:y+h, x:x+w])
    
# for face in face_crop:
#     cv2.imshow('face',face)
#     cv2.imwrite('cropped.jpg', face)
#     cv2.waitKey(0)
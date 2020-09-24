import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# dir = './img_lbp'

# data = []


# categories = ['humman1', 'humman2']
# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)

#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         humanImg = cv2.imread(imgpath, 0)
#         try:
#             humanImg = cv2.resize(humanImg, (50,50))
#             Image = np.array(humanImg).flatten() 
#             data.append([Image, label])
#         except Exception as e:
#             pass


# pick_in = open('data.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.98)
model =  SVC(C=1,kernel='poly', gamma= 'auto')
model.fit(xtrain, ytrain)

pick = open('model.sav', 'wb')
# model = pickle.load(pick)
pickle.dump(model, pick)
pick.close()

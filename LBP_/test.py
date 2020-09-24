import matplotlib.pyplot as plt
import cv2
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()


prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)
categories = ['human1', 'human2']

print('accuraacy: ', accuracy)
print('prediction is : ',categories[prediction[0]])

my = xtest[0].reshape(50,50)

plt.imshow(my, cmap='gray')
plt.show()

# img = cv2.imread("./test.jpg", 0)

# img_cv = cv2.resize(img, (185,185))
# image = np.array(img_cv).flatten()
# image = image.reshape(1,-1)
# pick = open('model.sav', 'rb')
# model = pickle.load(pick)
# pick.close()
# prediction = model.predict(image)

# print('prediction is:', max(prediction))
# # text = get


# cv2.imshow('ok', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

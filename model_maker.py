import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Activation


datadir = 'C:/Users/Vishaak/Downloads/face_mask/Dataset'
categories = ["without_mask","with_mask"]

img_size = 300

training_data = []

def create_train_data():
    
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        count = 0
        if(class_num == 0):
            for img in os.listdir(path):
                if(count<2800):
                    try:
                        img_array = cv2.imread(os.path.join(path,img))
                        new_array = cv2.resize(img_array,(img_size,img_size))
                        training_data.append([new_array,class_num])
                        count+=1
                    except Exception as e:
                        pass
        else:
            for img in os.listdir(path):
                if(count<2000):
                    try:
                        img_array = cv2.imread(os.path.join(path,img))
                        new_array = cv2.resize(img_array,(img_size,img_size))
                        training_data.append([new_array,class_num])
                        count+=1
                    except Exception as e:
                        pass
            
                

create_train_data()
print(len(training_data))
random.shuffle(training_data)


x = []
y = []

for feature,label in training_data:
    x.append(feature)
    y.append(label)

for i in range(len(x)):
    x[i] = x[i]/300

x = np.array(x)
y = np.array(y)

x = x.reshape(1,img_size,img_size,3)



#making the model

model = Sequential()

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

model.fit(x,y,batch_size = 10,validation_split = 0.2,epochs=3)

model.save("mask_detector.model")


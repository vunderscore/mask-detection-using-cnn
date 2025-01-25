import tensorflow as tf 
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('C:/Users/Vishaak/mask_detector.model')

test_path = os.path.join('C:/Users/Vishaak/Downloads/face_mask/Dataset/with_mask','2915.png')
test_img = cv2.imread(test_path)
plt.imshow(test_img)
plt.show()
test_img = cv2.resize(test_img, (300,300))
t_img = test_img.reshape(1,300,300,3)

predict = new_model.predict(t_img)
p_val = predict[0][0]
if(p_val == 1):
	print("mask detected")
else:
	print("no mask detected")
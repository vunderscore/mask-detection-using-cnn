import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("C:/Users/Vishaak/mask_detector_cnn.model")

cap = cv2.VideoCapture(0)

picture_flag = 0
picture_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX


while True:
	ret, frame = cap.read()

	width = int(cap.get(3))
	height = int(cap.get(4))

	frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	pass_image = cv2.resize(frame_bgr,(256,256))
	p_image = pass_image.reshape(-1,256,256,3)
	predict = model.predict(p_image)
	p_value = predict[0][0]


	flipped_frame = cv2.flip(frame,1)
	if(p_value==1):
		cv2.putText(flipped_frame,"sign detected",(50,50),font,1,(255,255,0),2,cv2.LINE_4)	
	cv2.imshow('video',flipped_frame)

	if(cv2.waitKey(1)==ord('q')):
		break

cap.release()
cv2.destroyAllWindows()


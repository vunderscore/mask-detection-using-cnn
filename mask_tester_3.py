import cv2
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import shutil

start_time = time.time()
time_flag = 0
count = 0

url = 'rtsp://admin:123456@192.168.1.206:554/Streaming/Channels/101'
cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('C:/Users/Vishaak/mask_detector_cnn.model')
img_size = 256

if(cap.isOpened() == False):
	print('error capturing camera')


mask_flag = 0
picture_count = 0
picture_save_path = 'D:/code/mask_detection/masked_images'

width = int(cap.get(3))
height = int(cap.get(4))
size = (width,height)

video_save_path = 'D:/code/mask_detection/mask_videos'
fourcc = cv2.VideoWriter.fourcc(*'MJPG')
video_before = cv2.VideoWriter(os.path.join(video_save_path,'before_detection.avi'),fourcc,10,size)
video_after = cv2.VideoWriter(os.path.join(video_save_path,'after_detection.avi'),fourcc,10,size)

event_directory = 'D:/code/mask_detection/event'

while True:
	ret, frame = cap.read()

	video_list = os.listdir(video_save_path)

	curr_time = time.time()
	calc_time = curr_time - start_time

	if(calc_time >= 25):
		start_time = curr_time
		count+= 1
		before_name = 'before_detection' + str(count) + '.avi'
		video_before = cv2.VideoWriter(os.path.join(video_save_path, before_name),fourcc,10,size)


	#frame = cv2.flip(frame,0)
	frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	frame_bgr = cv2.resize(frame_bgr, (img_size,img_size))
	p_image = frame_bgr.reshape(1,img_size,img_size,3)
	prediction = model.predict(p_image)
	p_value = prediction[0][0]

	if(p_value == 1):
		print('mask detected')
		picture_count+=1
		if(mask_flag == 0):
			cv2.imwrite(os.path.join(picture_save_path,'mask_detection_test.jpg'),frame)
			mask_flag = 1
			shutil.copy(os.path.join('D:/code/mask_detection/mask_videos',video_list[-1]),event_directory)
			shutil.copy(os.path.join('D:/code/mask_detection/mask_videos',video_list[-2]),event_directory)



	if(mask_flag == 1):
		video_after.write(frame)
	else:
		video_before.write(frame)
	

	cv2.imshow('video',frame)

	if(cv2.waitKey(1) == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
prog_time = end_time - start_time

shutil.copy(os.path.join(video_save_path,'after_detection.avi'),event_directory)


print("number of frames with mask:",picture_count)

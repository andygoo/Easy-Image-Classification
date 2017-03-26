import keras.preprocessing
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import img_to_array
import threading
from keras import backend as K	
import time
label = ''
frame = None
classes = ["ripe","semiripe","unripe"]

def create_model():
	json_file = open('/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_model','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_weights.h5")
	print("model is loaded")

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.stop = threading.Event()

	def run(self):
		global label
		# Load the VGG16 network
		self.create_model()
		while (~(frame is None)):
			label = self.predict(frame)

	def predict(self, frame):
		x = img_to_array(frame)
		x = np.rot90(x)
		x = x.reshape((1,) + x.shape) 
		fuckme=self.loaded_model.predict(x)
		print(fuckme)
		return classes[np.argmax(fuckme)]

	def terminate(self):
		self.stop.set()

	def create_model(self):
		json_file = open('/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_model','r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		self.loaded_model.load_weights("/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_weights.h5")
		print("model is loaded")




cap = cv2.VideoCapture(0)


keras_thread = MyThread()
keras_thread.start()




while (True):
	ret, original = cap.read()

	camera_capture = original[61:420, 0:640]  #0.56
	frame = cv2.resize(camera_capture, (169, 300)) 
	

	# Display the predictions
	cv2.putText(original, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", original)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
		break;

cap.release()
frame = None
cv2.destroyAllWindows()
keras_thread.terminate()
sys.exit()


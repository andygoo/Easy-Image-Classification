import cv2
import os
import time
import numpy as np
# Camera 0 is the integrated web cam on my netbook
# Camera 1 for logitech cam
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import img_to_array

	
json_file = open('/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_model','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/elements/Desktop/Easy-Image-classification/real_model/vgg16_weights.h5")
print("model is loaded")

loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

classes=["ripe","semi ripe","unripe"]


start_time = time.time()
cap = cv2.VideoCapture(0)


temp_remp, im = cap.read()

for i in range(10): #ramp frames
	temp = im
print("Saving Image....")

directory = "test_images"

if not os.path.exists(directory):
	os.makedirs(directory)

camera_capture = im
camera_capture = camera_capture[61:420, 0:640]  #0.56
camera_capture = cv2.resize(camera_capture, (169, 300)) 

x = img_to_array(camera_capture)
x = np.rot90(x)
x = x.reshape((1,) + x.shape) 
print()
fuckme=loaded_model.predict(x)


file = "Img_" + time.strftime("%d:%m:%y_date_") + time.strftime("%I:%M:%S_time") +".jpg"
print(classes[np.argmax(fuckme)])
print(fuckme)
cv2.putText(im, classes[np.argmax(fuckme)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(os.path.join(directory, file), im)

print("--- %s seconds ---" % (time.time() - start_time))






import h5py
import json
import keras
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K	
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.callbacks import History, TensorBoard, EarlyStopping

import os

import scikitplot as skp
import scikitplot.plotters as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits as load_data
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB

def evaluate(history_finetune, dir_loc):
	#ok first the retrain
	path = os.path.join(dir_loc,"Charts and Graphs")
	os.makedirs(path, exist_ok=True)

	#plot for accuracy validation and training
	plt.plot(history_finetune.history['acc'])
	plt.plot(history_finetune.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(path + '/accuracy.png')
	plt.close()
	
	#plot for losses validation and training
	plt.plot(history_finetune.history['loss'])
	plt.plot(history_finetune.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(path + '/loss.png')
	plt.close()


def vgg16(train, validate, 
		  losstype, nb_steps_toplayer, 
		  nb_steps_finetune, learning_rate, 
		  tbCallBack, early_stop, nb_classes, 
		  nb_training_samples, 
		  nb_validation_samples, 
		  dir_loc):
	from keras.applications.vgg16 import VGG16

	base_model = VGG16(weights='imagenet', include_top=False, input_shape=train.image_shape)

	if nb_classes > 2:
		gg = nb_classes
	elif nb_classes == 2:
		gg = 1

	x = base_model.output
	x = Flatten(input_shape=base_model.output_shape[1:])(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.7)(x)
	predictions = Dense(gg, activation='softmax')(x)

	model = Model(input=base_model.input, output=predictions)
	for layer in base_model.layers:
	    layer.trainable = False

	model.compile(optimizer='rmsprop', loss=losstype, metrics=['accuracy'])

	history_retraining = model.fit_generator(train, 
								steps_per_epoch = nb_training_samples, 
								epochs = nb_steps_toplayer, 
								verbose=1, 
								validation_data = validate, 
								validation_steps = nb_validation_samples)
	

	print("top model training done...")

	for layer in model.layers[:15]:
	   layer.trainable = False
	for layer in model.layers[15:]:
	   layer.trainable = True


	from keras.optimizers import SGD

	model.compile(optimizer=SGD(lr=learning_rate, momentum=0.8), loss=losstype, metrics=['accuracy'])


	print("finetuning starting...")
	history_finetuning = model.fit_generator(train, 
											steps_per_epoch = nb_training_samples, 
											epochs = nb_steps_finetune, 
											verbose=1, 
											validation_data = validate, 
											validation_steps = nb_validation_samples)
	
	print("finetuning_done")	

	probas_finetune = model.predict_generator(validate, nb_validation_samples)

	evaluate(history_finetuning, dir_loc)

	model_json = model.to_json()
	_ = os.path.join(dir_loc,"model and weights")
	os.makedirs(_, exist_ok=True)
	with open(os.path.join(_,"vgg16_model"), "w") as json_file:
	    json_file.write(model_json)
	model.save_weights(os.path.join(_,"vgg16_weights.h5"))
	print("Saved model")
	
def vgg19(train, validate, 
		  losstype, nb_steps_toplayer, 
		  nb_steps_finetune, learning_rate, 
		  tbCallBack, early_stop, nb_classes, 
		  nb_training_samples, 
		  nb_validation_samples, 
		  dir_loc):
	from keras.applications.vgg19 import VGG19

	base_model = VGG19(weights='imagenet', include_top=False, input_shape=train.image_shape)

	if nb_classes > 2:
		gg = nb_classes
	elif nb_classes == 2:
		gg = 1

	x = base_model.output
	x = Flatten(input_shape=base_model.output_shape[1:])(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(gg, activation='softmax')(x)

	model = Model(input=base_model.input, output=predictions)
	for layer in base_model.layers:
	    layer.trainable = False

	model.compile(optimizer='rmsprop', loss=losstype, metrics=['accuracy'])

	history_retraining = model.fit_generator(train, 
								steps_per_epoch = nb_training_samples, 
								epochs = nb_steps_toplayer, 
								verbose=1, 
								validation_data = validate, 
								validation_steps = nb_validation_samples)
	
	print("top model training done...")

	for layer in model.layers[:17]:
	   layer.trainable = False
	for layer in model.layers[17:]:
	   layer.trainable = True


	from keras.optimizers import SGD
	model.compile(optimizer=SGD(lr=learning_rate, momentum=0.8), loss=losstype, metrics=['accuracy'])


	print("finetuning starting...")
	history_finetuning = model.fit_generator(train, 
											steps_per_epoch = nb_training_samples, 
											epochs = nb_steps_finetune, 
											verbose=1, 
											validation_data = validate, 
											validation_steps = nb_validation_samples)
	print("finetuning_done")	

	probas_finetune = model.predict_generator(validate, nb_validation_samples)

	evaluate(history_finetuning, dir_loc)

	model_json = model.to_json()
	_ = os.path.join(dir_loc,"model and weights")
	os.makedirs(_, exist_ok=True)
	with open(os.path.join(_,"vgg19_model"), "w") as json_file:
	    json_file.write(model_json)
	model.save_weights(os.path.join(_,"vgg19_weights.h5"))
	print("Saved model")

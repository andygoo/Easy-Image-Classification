import os
import sys
import h5py
import json
import numpy as np 
from PIL import Image
import argparse

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K	
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.callbacks import History, TensorBoard, EarlyStopping

from models import vgg16, vgg19

def build_data(training_data_dir, 
			   validation_data_dir,
			   img_width, img_height,
			   batch, nb_classes, 
			   shuffle_flag, 
			   classmode):
	
	datagen = ImageDataGenerator(rescale=1./255)
	

	generator_train = datagen.flow_from_directory(
	        training_data_dir,
	        target_size=(img_width, img_height),
	        batch_size=batch,
	        class_mode=classmode,
	        shuffle=shuffle_flag)

	generator_validation = datagen.flow_from_directory(
	        validation_data_dir,
	        target_size=(img_width, img_height),
	        batch_size=batch,
	        class_mode=classmode,
	        shuffle=shuffle_flag)

	return generator_train, generator_validation


def train(model_label,
		  nb_steps_toplayer,
		  nb_steps_finetune, 
		  training_data_dir, 
		  validation_data_dir, 
		  img_width, img_height, 
		  batch, nb_classes, 
		  shuffle_flag, 
		  learning_rate,
		  nb_training_samples,
		  nb_validation_samples,
		  directory_location):
	
	os.makedirs(directory_location, exist_ok=True)

	if nb_classes == 2:
		classmode, losstype = "binary", "binary_crossentropy"
	elif nb_classes > 2:
		classmode, losstype = "categorical", "categorical_crossentropy"
		

	train, validate = build_data(training_data_dir, 
								 validation_data_dir, 
								 img_width, img_height, 
								 batch, nb_classes, shuffle_flag, classmode)

	early_stop = EarlyStopping(monitor='val_loss', patience=3)

	tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph_finetune', 
												 histogram_freq=0,
												 write_graph=True, 
												 write_images=True)
	
	if model_label == "vgg16":
		vgg16(train, validate, losstype, nb_steps_toplayer, nb_steps_finetune, learning_rate, tbCallBack, early_stop, nb_classes, nb_training_samples, nb_validation_samples, directory_location)

	elif model_label == "vgg19":
		vgg19(train, validate, losstype, nb_steps_toplayer, nb_steps_finetune, learning_rate, tbCallBack, early_stop, nb_classes, nb_training_samples, nb_validation_samples, directory_location)
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	      '--model_label',
	      type=str,
	      help='model inputs (vgg16, vgg19, inception_v3)')
	parser.add_argument(
	      '--nb_steps_toplayer',
	      type=int,
	      default='10',
	      help='number of steps you want to train top layer of the model only')
	parser.add_argument(
	      '--nb_steps_finetune',
	      type=int,
	      default='',
	      help='number of steps to finetune the trainable layers')
	parser.add_argument(
	      '--training_data_dir',
	      type=str,
	      default='',
	      help='training data directory')
	parser.add_argument(
	      '--validation_data_dir',
	      type=str,
	      default='',
	      help='validation data directory')
	parser.add_argument(
	      '--test_data_dir',
	      type=str,
	      default='',
	      help='testing data directory')
	parser.add_argument(
	      '--img_width',
	      type=int,
	      default='150',
	      help='location of the training directory default width = 150')
	parser.add_argument(
	      '--img_height',
	      type=int,
	      default='150',
	      help='location of the validation directory default height = 150')
	parser.add_argument(
	      '--batch_size',
	      type=int,
	      default='32',
	      help='size of image batches to be fed into generator')
	parser.add_argument(
	      '--nb_classes',
	      type=int,
	      help='number of distict classes')
	parser.add_argument(
	      '--shuffle_flag',
	      type=str,
	      default='False',
	      help='shuffling of images in generator')
	parser.add_argument(
	      '--learning_rate',
	      type=float,
	      default='0.0001',
	      help='learning rate for finetuning'
	  )
	parser.add_argument(
	      '--nb_training_samples',
	      type=int,
	      help='number of training sample inputs'
	  )
	parser.add_argument(
	      '--nb_validation_samples',
	      type=int,
	      help='number of validation sample inputs'
	  )
	parser.add_argument(
	      '--directory_location',
	      type=str,
	      default="/tmp",
	      help='location to save all graphs and files'
	  )
	args = parser.parse_args()



	train(args.model_label,
		  args.nb_steps_toplayer, 
		  args.nb_steps_finetune,
		  args.training_data_dir, 
		  args.validation_data_dir, 
		  args.img_width, args.img_height, 
		  args.batch_size, args.nb_classes, 		
		  args.shuffle_flag,
		  args.learning_rate,
		  args.nb_training_samples,
		  args.nb_validation_samples,
		  args.directory_location)




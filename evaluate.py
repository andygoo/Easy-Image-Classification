import numpy as np
import scikitplot as skp
import scikitplot.plotters as skplt
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import json
import os
import h5py
import numpy as np
from keras import backend as K
from sklearn.naive_bayes import GaussianNB

import argparse



parser = argparse.ArgumentParser()

parser.add_argument(
      '--model_location',
      type=str,
      help='location of model')
parser.add_argument(
      '--weights_location',
      type=str,
      help='location of saved weights')
parser.add_argument(
      '--testdir',
      type=str,
      help='direcory of images to evaluate')
parser.add_argument(
      '--destdir',
      type=str,
      help='directory where to save the images')
parser.add_argument(
      '--nb_classes',
      type=int,
      help='number of classes')
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



args = parser.parse_args()
json_file = open(args.model_location,'r')
loaded_model_json = json_file.read()
json_file.close()

with tf.device('/cpu:0'):
	args = parser.parse_args()

	json_file = open(args.model_location,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights(args.weights_location)
	print("model is loaded")

	if args.nb_classes == 2:
		classmode, losstype = "binary", "binary_crossentropy"
	elif args.nb_classes > 2:
		classmode, losstype = "categorical", "categorical_crossentropy"

	loaded_model.compile(loss=losstype, optimizer='rmsprop', metrics=['accuracy'])

	

	datagen = ImageDataGenerator(rescale=1./255)

	generator_test = datagen.flow_from_directory(
		args.testdir,
		target_size=(args.img_width, args.img_height),
		batch_size=1,
		class_mode=classmode,
		shuffle=False)

	probos_finetune=loaded_model.predict_generator(generator_test, len(generator_test.classes), max_q_size=1)
	gnd_truth = generator_test.classes
	
	dics = list(generator_test.class_indices.keys())

	dics.extend(["micro-average curve","macro-average curve"])

	

	skplt.plot_precision_recall_curve(y_true=gnd_truth, y_probas=probos_finetune)
	plt.legend(dics[:5], loc='lower left')
	plt.savefig(args.destdir + '/precision_recall_curve.png')
	plt.close()

	_ = classifier_factory(RandomForestClassifier())
	_.plot_confusion_matrix(probos_finetune, gnd_truth, normalize=True)
	plt.savefig(args.destdir + '/confusion-matrix.png')
	plt.close()

	nb = GaussianNB()
	classifier_factory(nb)
	nb.plot_roc_curve(probos_finetune, gnd_truth, title='roc curves')
	plt.legend(dics, loc='lower right')
	plt.savefig(args.destdir + '/ROC.png')
	plt.close()

	_.plot_learning_curve(probos_finetune, gnd_truth)
	plt.savefig(args.destdir + '/learning_curve.png')
	plt.close()



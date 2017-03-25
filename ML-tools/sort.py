import os
import random
import glob
import numpy
import shutil
import argparse

random.seed(7)

def listfiles(src_dir):
	files = []
	#test/
	for f in os.listdir(src_dir):
	    filename = os.path.join(src_dir,f)
	    
	    _ = glob.glob(filename + "/*jpg")
	    files.extend(_)
	    
	files_num = numpy.array(files)
	random.shuffle(files_num)
	return(files_num, len(files_num))

def main(srcdir, destdir, per_train, per_validate):

	hello, total = listfiles(srcdir)

	training = per_train / 100
	validation = per_validate / 100

	training_r = int(total * training)
	validation_r = training_r + int(total * validation)
	training, validation, test = hello[:training_r], hello[training_r:validation_r], hello[validation_r:]
	data = [training, validation, test]



	base_dir = ["training", "validation", "testing"]

	for x in base_dir:

	    #makes train, valid, test directories
	    tmp_dir = os.path.join(destdir,x)
	    if os.path.exists(tmp_dir) is False:
	        os.mkdir(tmp_dir)
	    for _ in data[base_dir.index(x)]:
	        base, pls = os.path.split(_)
	        base = os.path.basename(base)
	        if os.path.exists(os.path.join(tmp_dir, base)) is False:
	            os.mkdir(os.path.join(tmp_dir, base))
	        shutil.copy(_, os.path.join(tmp_dir, base))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument(
	      '--srcdir',
	      type=str,
	      help='location of directory')
	parser.add_argument(
	      '--destdir',
	      type=str,
	      help='directory to send the folder')
	parser.add_argument(
	      '--per_train',
	      type=int,
	      default='70',
	      help='percent of images sent into training folder ex: --per_train 70')
	parser.add_argument(
	      '--per_validate',
	      type=int,
	      default='20',
	      help='percent of images sent into validation folder ex: --per_validate 20')
	args = parser.parse_args()

	main(args.srcdir, args.destdir, args.per_train, args.per_validate)

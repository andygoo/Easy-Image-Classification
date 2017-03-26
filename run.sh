#!/bin/bash



python3 totalshit.py --model_label vgg19 \
		     --nb_steps_toplayer 10 \
		     --nb_steps_finetune 20 \
		     --training_data_dir "mango_dataset/training" \
		     --validation_data_dir "mango_dataset/validation" \
		     --nb_classes 3 \
		     --img_width 169 \
		     --img_height 300 \
		     --learning_rate 0.0001 \
		     --batch_size 1 \
		     --nb_training_samples 639 \
		     --nb_validation_samples 170 \
		     --directory_location "/home/elements/Desktop/inception_v3" \


python3 evaluate.py --model_location /home/elements/Desktop/test/model\ and\ weights/vgg16_model \
		    --weights_location /home/elements/Desktop/test/model\ and\ weights/vgg16_weights.h5 \
		    --testdir mango_dataset/validation/ \
		    --destdir /home/elements/Desktop/test/Charts\ and\ Graphs/ \
		    --nb_classes 3 --img_width 300 --img_height 300 \







			






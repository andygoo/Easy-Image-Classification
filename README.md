

# Easy Image Classification

This is to help thesis groups to easily train, evaluate and implement a Image classifier by finetuning on vgg16 or vgg19 models.

## Dependencies

## Training

Create a dataset of images on which you would retrain the model with. In my instance I am working with a group to implement a mango sorting system that is reliant on computer vision. Supposedly we need to be able to classify ripness and quality. Mango Dataset was provided by Dr. Edwin Calilung and his associates in Department of Agriculture Philmech. 

Images should be contained in their respective class folders.
Your images should be contained in a folder split between training and validation. I just always go with the 80:20 ratio. you can also add a seperate testing directory to evaluate the model after training. 

I also uploaded my own script in ML-tools folder it helps to randomly sort and prepare my training, validation and testing dataset.

```shell
python3 sort --srcdir "/images" #folder where images are seperated into classes
	     --destdir "/dataset" 
	     --per_train 70 #percentage of data split into training directory
	     --per_validate 20 #percentage of data split into validation directory
```
You're dataset should have a format as the one seen below.

```shell
  $TRAIN_DIR/dog/image0.jpeg
  $TRAIN_DIR/dog/image1.jpg
  $TRAIN_DIR/dog/image2.png
  ...
  $TRAIN_DIR/cat/weird-image.jpeg
  $TRAIN_DIR/cat/my-image.jpeg
  $TRAIN_DIR/cat/my-image.JPG
  ...
  $VALIDATION_DIR/dog/imageA.jpeg
  $VALIDATION_DIR/dog/imageB.jpg
  $VALIDATION_DIR/dog/imageC.png
  ...
  $VALIDATION_DIR/cat/weird-image.PNG
  $VALIDATION_DIR/cat/that-image.jpg
  $VALIDATION_DIR/cat/cat.JPG
  ...
```
Finally to train the dataset I made a program where it will help me easily retrain and finetune the vgg16 and vgg19
models. Finetuning only occurs up to the last convolutional block for each model. 

This code took a lot of inspiration from Francois Chollet [blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), read the blog to learn more about how finetuning was implemented 
here. So just like in his blog, the code goes through 2 steps. First is training the sequential dense layers and 
then the second step is to finetune the last convolutional block with the last few layers.    

```shell
#Example Usage, you can change these parameters however you want
python3 train.py --model_label vgg16 \ 
		     	 --nb_steps_toplayer 20 \ 
		     	 --nb_steps_finetune 50 \
		     	 --training_data_dir "mango_dataset/training" \
		     	 --validation_data_dir "mango_dataset/validation" \
		     	 --nb_classes 3 \
		     	 --img_width 300 \
		     	 --img_height 300 \
		     	 --learning_rate 0.0001 \
		     	 --batch_size 1 \
		     	 --nb_training_samples 3000 \
		       --nb_validation_samples 600 \
		       --directory_location "/home/elements/Desktop" \ #directory to save the ouput
```
So if you are starting this script for the first time, it will take a while due to having to download the model weights 
at the beginning. Training may be a bit slower maybe due to not caching the bottlneck features, I am not a 100% on 
this. But depending on your dataset it is still better than training a CNN model from scratch.

After training you should see two directories inside the directory specified. Directory names should be "Charts and 
Graphs" and "Model and Weights". So inside these directories I included the model, its weights and graph visualization 
for the training accuracy and loss.

![training accuracy and loss](https://github.com/ryanliwag/Easy-Image-Classification/blob/master/images/model_training.png)

## Evaluation

  Next step is evaluating the model. Simply specify where the trained model and weights are located, which were all produced by the running the train.py. Also specify which dataset you would want to evaluate your model over, you can use the previous validation or prepare a seperate testing dataset.


```shell
#Example Usage of the evaluate.py script

python3 evaluate.py --model_location vgg16_model 
          --weights_location vgg16_weights.h5 
          --testdir mango_dataset/testing/ 
          --destdir /home/evaluate 
          --nb_classes 3 --img_width 300 --img_height 300

```
The output figures should then be saved in your specified destination directory. Also if you're training with gpu and you have enough vram memory, you can comment out the line "with tf.device('/cpu:0'):" in evaluate.py.

![graphs and figures](https://github.com/ryanliwag/Easy-Image-Classification/blob/master/images/evaluate.png)

If you also want to visualize what your images looks like as it passes through the convolutional blocks. You can use [Quiver](https://github.com/keplr-io/quiver), great awesome interactive tool for visualizing covnets and very easy to implement with keras.

![covnets visualization](https://github.com/ryanliwag/Easy-Image-Classification/blob/master/images/covnet_sample.png)

## Implementation with OpenCV 


## Credits

## still updating

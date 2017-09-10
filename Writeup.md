#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center driving"
[image2]: ./examples/counterclock.jpg "Driving counter-clockwise"
[image3]: ./examples/side1.jpg "Recovery Image: initial position"
[image4]: ./examples/side2.jpg "Recovery Image: start to drive back to center"
[image4]: ./examples/side3.jpg "Recovery Image: recover down!"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of three convolution neural networks with 5x5 filter sizes and depths between 24 36 and 48 (model.py lines 59-61) 

My model also consists of two convolution neural networks with 3x3 filter sizes and depths of 64 (model.py lines 62-63) 

The model includes RELU layers to introduce nonlinearity (code line 59-63), and the data is normalized in the model using a Keras lambda layer (code line 56). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 70).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to analyze different popular neural network models and see which one works best.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it is very classical and straightforward.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model similar to the famous model for self-driving from NVIDIA.

Then I repeat the whole training and validation process. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 55-68) consisted of convolution neural networks with the following sizes:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5     	| output depth: 24 	|
| RELU					|												|
| Convolution 5x5     	| output depth: 36 	|
| RELU					|												|
| Convolution 5x5     	| output depth: 48 	|
| RELU					|												|
| Convolution 3x3     	| output depth: 64 	|
| RELU					|												|
| Convolution 3x3     	| output depth: 64 	|
| RELU					|												|
| Flatten |
| Fully connected		| Dense(100)|
| Fully connected		| Dense(50)|
| Fully connected		| Dense(10)|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I also record the data of driving counter-clockwise for more training data.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from abnormal driving status.  These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generate more data for training.

After the collection process, I had 1956 number of data points. I then preprocessed this data by normalization using Lambda(model.py lines 56)

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the result of simulator.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

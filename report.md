# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Record a video of the successful drive
* Summarize the results with a written report

## Rubric Points
Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network 
* this report.md - summarizing the results
* video.mp4 - recording of the autonomous mode of the vehicle

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
source activate carnd-term1
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

#### Design Approach

The overall strategy for deriving a model architecture was to chop irrelevant part of the environment and squeeze out the most out of the images by applying 5 convolutional layers with different filter kernels and paddings.

I have re-used the CNN architecture, designed by [Nvidia)(https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

Here is a visualization of the architecture:

<img src="/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/cnn-architecture-624x890.png" style="zoom: 50%;" />

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set. 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced the number of epochs to 5.

#### Model Architecture in Detail

The final model architecture (model.py lines 89-110) consist of preprocessing, data processing, trainging and saving the model.

##### Preprocessing

1. Images are normalized in the model using a Keras lambda layer (model.py line 89) 
2. Top and bottom of the images are chopped to remove noise of the environment (top) and a front of a car (bottom)

##### CNN Architecture

My model consists of a 3 convolutional layers with 5x5 and 2 convolutional layers with 3x3 filter sizes and depths between 24 and 64 (model.py lines 93-98) 

The model includes ReLU activation function in the first three convolutional layers to introduce nonlinearity (model.py lines 93-95).

In the 6th layer the model is flattened.

Last 4 fully-connected layers reduce the dencity of the model from 100 to 1 neurons.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).



### Training Strategy

#### Training Data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:

*  perfect_line: center lane driving

* recovery_left: recovering from the left side of the road. 

* recovery_right: recovering from the right side of the road.

  Since images occupy roughly 700Mb of space, they are not uploaded. Let me know if you need them.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_13_21_59_955.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to get back to the perfect line without knowing on how to get off the perfect line. These images show what a recovery looks like:

![](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_14_05_19_070.jpg)

![center_2020_06_07_14_05_19_896](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_14_05_19_896.jpg)

![center_2020_06_07_14_05_20_933](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_14_05_20_933.jpg)

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_13_42_03_824.jpg)

![center_2020_06_07_13_42_03_824_flip](/home/q426889/priv_repo_mboiko/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/examples/center_2020_06_07_13_42_03_824_flip.jpg)

After the collection process, I had 17682 number of data points. In total, including center, left and right camera images and flipping all of those, the model had bee trained with 84874 images and validated with 21218 images.

#### Training Process

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation loss after every epoch run. I used an adam optimizer so that manually training the learning rate wasn't necessary.

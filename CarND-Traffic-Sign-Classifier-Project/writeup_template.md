# **Traffic Sign Recognition** 

## Introduction
This is the project of udacity Term1 German traffic sign classifier project. The goal of this project is the following:
 
* Load the data set (see below for links to the project data set)
* Data preprocessing, Data augmentation
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/test_images_from_google/00_german-road-sign-road-narrows-both-sides-narrow-bottleneck-construction-J2HT7X.jpg "Both_narrow"
[image2]: ./examples/test_images_from_google/01_roadsign_yield-e1452287501390.png "Yield"
[image3]: ./examples/test_images_from_google/02_work-progress-road-sign-triangle-isolated-cloudy-background-germany-47409527.jpg "Work_progress"
[image4]: ./examples/test_images_from_google/03-Speed-limit-sign-in-Germany-Stock-Photo.jpg "Speed_limit"
[image5]: ./examples/test_images_from_google/04_sign-giving-order-no-entry-vehicular-traffic.jpg "No_entry"
[image6]: ./examples/test_images_from_google/05_german-road-sign-bicycles-crossing-j2mra8.jpg "Bicycle_crossing"
[image7]: ./examples/random_train_sample.png "Random_train_sample"
[image8]: ./examples/train_hist.png "Train_histogram"
[image9]: ./examples/preprocess_img.png "Preprocessed image"
[image10]: ./examples/transform_img.png "Transformed image"

---
## Dataset
The dataset consists of training, validation and test data. Each images has 32x32x3 pixels and RGB colorspace. The number of example  data is following:

| Dataset | Shape of examples | Classes |
|---|---:|:---:|
| Training | 34,799 x 32 x 32 x 3 | 43 |
| Validation | 4,410 x 32 x 32 x 3 | 43 |
| Test | 12,630 x 32 x 32 x 3 | 43 |

This is the 43-class names of the dataset. I used pandas module to read csv file.

| Class ID | SignalName | Class ID | SignalName |
|---|:---|---|:---|
|	0 |	Speed limit (20km/h) | 22 | Bumpy road |
|	1 |	Speed limit (30km/h) | 23 | Slippery road |
|	2 |	Speed limit (50km/h) | 24 | Road narrows on the right |
|	3 |	Speed limit (60km/h) | 25 | Road work |
|	4 |	Speed limit (70km/h) | 26 | Traffic signals |
|	5 |	Speed limit (80km/h) | 27 | Pedestrians |
|	6 |	End of speed limit (80km/h) | 28 | Children crossing |
|	7 |	Speed limit (100km/h) | 29 | Bicycles crossing |
|	8 |	Speed limit (120km/h) | 30 | Beware of ice/snow |
|	9 |	No passing | 31 | Wild animals crossing |
|	10 |	No passing for vehicles over 3.5 metric tons | 32 | End of all speed and passing limits |
|	11 |	Right-of-way at the next intersection | 33 | Turn right ahead |
|	12 |	Priority road | 34 | Turn left ahead |
|	13 |	Yield | 35 | Ahead only |
|	14 |	Stop | 36 | Go straight or right |
|	15 |	No vehicles | 37 | Go straight or left |
|	16 |	Vehicles over 3.5 metric tons prohibited | 38 | Keep right |
|	17 |	No entry | 39 | Keep left |
|	18 |	General caution | 40 | Roundabout mandatory |
|	19 |	Dangerous curve to the left | 41 | End of no passing |
|	20 |	Dangerous curve to the right | 42 | End of no passing by vehicles over 3.5 metric ... |
|	21 |	Double curve | 43 |  |

### Dataset Exploration
I used matplotlib to visualize the images in the notebook. I selected 25 samples randomly from entire training samples.

![alt text][image7]

and dataset distributions are ploted below. It's very unbalanced data.(Minimum 130 to Maximum 2010)

![alt_text][image8]

### Preprocessing
As we seen the dataset exploration, Images has uneven brightness. So we need to pre-process the images. First, I convert RGB color space to gray, and normalize it with mean zero and equal variance approximately. Last, I apply adaptive histogram normalization(CLAHE) to solve the unbalanced brightness.

![alt_text][image9]

As you can see above, images brightness are equalized.

### Augmentation
The number of training dataset is relatively enough to train. but some class has very small number of data to generalize the model. I use image augmentation techniques to generate enough dataset. Random rotation, translation and shearing are used. After data augmentation, training sample are increased to x10 time(347,990 samples). The test transformed images are follwed: (1,1) is original image

![alt_text][image10]

---
## Model
### Architecture
### Training
### Results

---
## Test
### New Images
### Performance on New Images
### Top 5 results

---
## Conclusion


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



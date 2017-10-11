# **Traffic Sign Recognition** 

## Introduction
This is the project of udacity Term1 German traffic sign classifier project. The goal of this project is the following:
 
* Load the data set (see below for links to the project data set)
* Data preprocessing, Data augmentation
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is a link to my [project code](https://github.com/jaeoh2/self-driving-car-nd/blob/master/CarND-Traffic-Sign-Classifier-Project-P2/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./examples/test_result5.png "Both_narrow"
[image2]: ./examples/test_result3.png "Yield"
[image3]: ./examples/test_result2.png "Work_progress"
[image4]: ./examples/test_result6.png "Speed_limit"
[image5]: ./examples/test_result4.png "No_entry"
[image6]: ./examples/test_result.png "Bicycle_crossing"
[image7]: ./examples/random_train_sample.png "Random_train_sample"
[image8]: ./examples/train_hist.png "Train_histogram"
[image9]: ./examples/preprocess_img.png "Preprocessed image"
[image10]: ./examples/transform_img.png "Transformed image"
[image11]: ./examples/layers.png "Model layers"
[image12]: ./examples/train_result.png "Train result"
[image13]: ./examples/perf_result.png "Performance result"

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
As we seen the dataset exploration, Images has uneven brightness. So we need to pre-process the images. First, I convert RGB color space to gray, and normalize it with mean zero and equal variance approximately. Last, I applied adaptive histogram normalization(CLAHE) to solve the unbalanced brightness.

![alt_text][image9]

The difference between images are above. As you can see, the images brightness were equalized.

### Augmentation
The number of training dataset is relatively enough to train. but some class has very small number of data to generalize the model. I used image augmentation techniques to generate enough dataset. Random rotation, translation and shearing were used. After data augmentation, training sample are increased to x10 time(347,990 samples). The test transformed images are follwed: (1,1) is original image

![alt_text][image10]

---
## Model
I used common CNN classifier structure(CNN after FC). CNN layers are similar with VGG. It has 3 CNN layers. Fully connected layer has 2-hidden layers. Activation function used 'elu'. Dropout are used in FC, Batch Normalization are in CNN. Optimizer is stocastic gradient descent and cost function is softmax. 
### Architecture
The model architectures are below :

![alt_text][image11]

### Training
Train on 347,990 samples, validate on 4,410 samples. Batch size is 1024 per GPU(I use two Tesla K80 GPU), epochs 100.
Early stopping strategy was used. Learning rate is started from 1e-2 and it decayed size of 1e-6. Training finished at 30 epoch with loss : 0.0092, acc: 0.9973 / val_loss: 0.1359, val_acc: 0.9794.

![alt_text][image12]

### Results
The final results of my model were :
* accuracy of the test dataset is __94.39%__
* accuracy of the train dataset is 99.73%
* accuracy of the validation dataset is 97.94%

At first, I think this is the common image classification problem. But the number os datasets were small to train well. When I tested same CNN model architecture without data preprocessing, I got about 80%. So I decided to focus on making the preprocessing dataset. If we have enough numbers of dataset for image classification problem, we can get high accuracy model even using simple CNN architecture. This is why I spend more time to make data augmentation features then modified the model for this project.

---
## Test a Model on New Images
Here are the test result of 6 German traffic signs that I found on the web:

![alt_text][image1] ![alt_text][image2] ![alt_text][image3] 
![alt_text][image4] ![alt_text][image5] ![alt_text][image6] 

### Performance on New Images
The performance on first 5 images are very clear to recognize it. The test images located very well in the center of images. But last one, the speed limit data is biased and relatively smaller than first 5 samples. So, during the resize the images to 32x32 pixels, I think speed limit number informations were losted. This is the reason mis-classified result of speed limit data.

### Top 5 results
Here are the top-5 result of the Test images from web:

![alt_text][image13]

---
## Conclusion
Using German traffic sign dataset and simple CNN model, we get traffic sign classifier with accuracy of more then 94%. I applied data preprocessing(rgb2gray, histogram equalization) and data augmentation strategy(random rotate, translate and shear). To improve the accuracy of the model, Im considering to modify the CNN model and research for data balancing approch. 

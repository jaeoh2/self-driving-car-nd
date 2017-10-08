# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/hist.png "Histogram"
[image2]: ./examples/histbal.png "Balanced histogram"
[image3]: ./examples/modelsummary.png "Model Summary"
[image4]: ./examples/transformed.png "Transformed Image"

## Introduction
**Behavioral Cloning Project-3**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Here is the links to my [project code](https://github.com/jaeoh2/self-driving-car-nd/blob/master/CarND-Behavioral-Cloning-P3/Behavioral-Cloning.ipynb)

---

## Dataset
The dataset collected from [Udacity Self-driving simulator](https://github.com/udacity/self-driving-car-sim). The train data collected from each track A and B courses in 3-laps. Each images has 160x320x3 pixels and has 3 different camera viewpoint as center,left and right. For my train data has 13,977 examples and it augmented to 258,600 training samples.
### Dataset Exploration
![alt_text][image1]

The collected dataset is biased to 0 deg steering angle state. I applied data balancing approach to my dataset.
![alt_text][image2]

### Augmentation
![alt_text][image4]
#### Random brightness
#### Random flip image

### Preprocessing
Cropping, Resizing and RGB color spaces
 * plt cropping and resizing images

---

## Model
I used Nvidia's End to End self driving CNN models in this projects. It has 3 CNN layers and 3 Fully connected hidden layes. Activation function used 'elu'. Dropout are used in FC layes to prevent overfit. The optimizer is adam and the cost function is mean-squared-error.

### Architecture
The model architectures are below:
![alt_text][image3]

### Training
Train on 167,720 samples, validate on 8,387. Batch size is 128, epochs are 10. Early stopping callbacks applied. 

## Results
 | [![Alt text](https://img.youtube.com/vi/BMOCWUwIXKc/0.jpg)](https://youtu.be/BMOCWUwIXKc) |
 |:--:|
 | *Track 1* |
 | [![Alt text](https://img.youtube.com/vi/85vzNDCZT78/0.jpg)](https://youtu.be/85vzNDCZT78) |
 | *Track 2* |
 
### Notes
 * At first, I applied keras generator function to model training. But it was extremely slow to train, I changed it to saving datas to memory. I think when generator transmit the datas to GPU there was some bottleneck on network speeds because the train data was in NAS.
 * I collected the train data in track A only. It was not bad in track A but the car ran into the cliff in track B. I tried various data augmentation techniques, It was no differences. I collected only 3-laps from track B and the cars start to drive. I think it was much better to collect more data than data augmentation.
 * Use the analog joy-sticks, at least use the mouse steering.
 * Tested color spaces(YUV, HSV) but RGB was the best for my cases.
 * Random shadow or ramdom brightness approaches was not much effects to get better performance.

# Project1 : **Finding Lane Lines on the Road**
## Introduction
Detecting lane lines on the Road using computer vison tools like OpenCV, python and Hough transform algorithm.
The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Describe a project pipeline and results.

## Pipeline description
My lane detection pipeline has 4-steps are below:
1. Pre-Process images : Converting gray scale, gaussian Blurring and canny transform.
1. Set Region of Interests in polygon shape.
1. Hough transform
1. Draw lane lines overlayed on original image
* To Draw lane lines as solid line, I stored entire x,y positions gets from hough transform result to lists.
and calculate each line slope seperately. Based on average slope and bias values, I can estimate
end points in the image with simple 1st order line fomula. Saturating the x,y positions within ROI boundary,
I can get straight line on the each lane.

## Shortcomings
### Only detect on straight line
My pipeline can draw the straight line only. Beacuse draw line function has 1st order line formula. If there is corners in the image, average line slope will be biased. 
Results image will add

## Suggest improvements
### Curve drawing
As described in shortcomings, If we can plot the lane line features to curve fomula, we can draw the lane line without biased.
### Different colorspace
Currently we use only gray color space. Especially, the road has limited lane colors like white, yello and etc.
So, I think we may get little improvements from different colorspace. 
### Histogram Normalization
Since there are different brightness environment on the road (shade, night, tunnel, etc), we need to normalize the
images pixel intensities before vision processing. Using adaptive hitogram equalization algorithm like CLAHE, we 
can get better detection result from same pipeline.

## Conclusions
Using simple computer vision algorithm (canny, hough transform), we can easily get the lane detection program.
But it's not enough to adapt in real situations. In different roads, weathers and brightness, there are chance to fail
detecting lane lines. To come up with this shortcoming, we can use histogram normalization adaptively in various situation.

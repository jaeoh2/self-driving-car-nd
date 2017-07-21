# Project1 : **Finding Lane Lines on the Road**
## Introduction
Detecting lane lines on the Road using computer vison tools like OpenCV, python and Hough transform algorithm.
The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Describe a project pipeline and results.

## Pipeline description
My lane detection pipeline has 5-steps are below:
1. Pre-Process images : Converting gray scale, gaussian Blurring and canny transform.
1. Set Region of Interests in polygon shape.
1. Hough transform
1. Draw lane lines overlayed on original image
* To Draw lane lines as solid line, I stored entire x,y positions gets from hough transform result to lists.
and calculate each line slope seperately. Based on average slope and bias values, I can estimate
end points in the image with simple 1st order line fomula. Saturating the x,y positions within ROI boundary,
I can get straight line on the lane.

## Shortcomings
One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...

## Suggest improvements
A possible improvement would be to ...

Another potential improvement could be to ...

## Conclusions

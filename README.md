**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Output/1.png
[image2]: ./Output/2.png
[image2_2]: ./Output/3.png
[image2_3]: ./Output/4.png
[image2_4]: ./Output/5.png

[image5]: ./Output/6.png
[image6]: ./Output/7.png
[image7]: ./Output/8.png

[image8]: ./Output/9.png
[image9]: ./Output/grey.png
[image10]: ./Output/10.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `main.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images. To enhance the quality of the SVC model, 4 sets of data has been used, which is `non-vehicles`, `non-vehicles_smallset`, `vehicles`and`vehicles_smallset`.To avoid overfiting, a denominator = 5 has been used in line 35-42 of cell 2. The total amounts of 1998 cars and 2019 non-cars has been used as training set.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image2_2]
![alt text][image2_3]
![alt text][image2_4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finaly the discussion in [Good tips from my reviewer for this Vehicle Detection Project](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903/11) helps me improved the detection quality.
The following is my final parameters:
```
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial = 32
histbin = 32
spatial_size = (32, 32) # Spatial binning dimensions
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC` in line 52-55 of cell 5. The parameters of the training HOG features is shown in above.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at 10 scales all over the image in cell 9:
![alt text][image8]

|No.|Y Start-Stop     		|     Scale	     		| 
|:-:|:---------------------:|:---------------------:| 
| 1 |400-500        		|1.0 					| 
| 2 |400-500        		|1.3 					| 
| 3 |410-500        		|1.4 					| 
| 4 |420-556         		|1.6 					| 
| 5 |430-556        		|1.8 					| 
| 6 |430-556        		|2.0 					| 
| 7 |440-556        		|1.9 					| 
| 8 |400-556        		|1.3 					| 
| 9 |400-556        		|2.2 					| 
| 10|500-656        		|3.0 					| 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 10 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:
![alt text][image5]

---

### Video Implementation

#### 1. Video link to your final video output. 
Here's a [link to my video result](./Output/project_svm.mp4)
[![project_svm](https://img.youtube.com/vi/c3Q7o7TchPQ/0.jpg)](https://www.youtube.com/watch?v=c3Q7o7TchPQ)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.(Pipeline line 47-53)  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some frames and their corresponding heatmaps:
![alt text][image5]
![alt text][image6]
![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image9]

### Here the resulting bounding boxes are drawn onto one frame in the series:
![alt text][image10]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* To enhance the quality of the model, I push two packages of data into training set. The type of the data is jpg and png. Since the `mpimg.imread` would take png as [0,1], so I enlarge the png picture by * 255:
```
        if 'png' in file:
            image = mpimg.imread(file)
            image = image.astype(np.float32)*255
        else:
            image = mpimg.imread(file)
```
line 44-48 in cell 4. In this way, all the image would be training as [0,255] data range and so as the frame take from the video.
* The interesting part of the project is to adjust the parameters when training the svc model. I first failed with `RGB`, `HSV` colorspace and (16,16) spatial size.

#README

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the requirements individually and describe how I addressed each one in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The `main()` function tries to load the dataset from a pickled file `car-no-car.p`. If it doesn't find it, then it loads the dataset images, and pickles them along with their labels calling `load_and_pickle_datasets()`. The dataset has been obtained from:
 
 * GTI vehicle image database
 * KITTI vision benchmark suite
 * Udacity's annotated driving dataset
 
`main()` shuffles and splits the dataset between a training set (90%) and a validation set (10%) calling `shuffle_and_split_dataset()`. It then calls `fit_and_pickle_classifier()`, which extracts features from the dataset images before fitting a classifier. 

I adopted OpenCV implementation of HOG, `cv2.HOGDescriptor()`, finding it faster than scikit-image implementation, `skimage.feature.hog()`. It also allows tuning some additional parameters.

Method `get_hog_descriptor()` initializes parameters for the HOG, and method `compute_hog_features()` extracts and returns HOG features for a given image. It can operate on a single channel image, or a 3 channels image.

Method `compute_histogram_features()` returns concatenated histogram features for a given image and choice of image channels.
 
Function `compute_image_features()` concatenates and returns HOG and color histogram features for a given image, not scaled. After numerous experiments, I settled for a features descriptor that concatenates:
  * a HOG for the V channel of the image in HSV color space;
  * a histogram (16 bins each) for the U and V channels of the image in YUV color space.

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Images, before being used to train and validate the classifier and for detection, are all resized to 64x64 pixels.

To compute the HOG, I used a cell size of 8x8 pixels, a block size of 2x2 cells, a stride for blocks of 8x8 pixels (i.e. 1 cell), 16 bins, gamma correction and signed gradient.

I started with 9 bins and unsigned gradient, based on the paper about pedestrians detection, but then found that increasing the number of bins to 16 and adopting a signed gradient increased drastically the detection of true positives in video frames. 

I had three ways to test and tune parameter values, for HOG and other stages of the pipeline: testing the classifier on a validation dataset, testing detection on selected camera frames, and testing detection on the whole input video clip. The latter is the most time consuming, and could only be done after the pipeline was completely implemented. 
  
I tuned parameters by trial and error first running the trained classifier on a validation dataset I selected,  until further tweaking didn't bring about significant improvement, and then on frames I selected from the input video clip. Test on video frames was subjective.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


Method `fit_and_pickle_classifier()` re-scales image features with `sklearn.preprocessing.StandardScaler`, which results in improved detection accuracy. It then fits a Support Vector Machine (SVM) linear classifier with `sklearn.svm.LinearSVC¶()`, evaluates the result on the validation dataset, and saves scaler and classifier in a pickled file. At the next run, it will load them from the pickled file and avoid repeating the computation.
 
 I set the C parameter for the Linear SVM classifier to 0.001 by trial and error, based on the classifier performance on the validation set. I didn't find a very fine tuning worthwhile, perhaps based on a grid search. Comparing performance on the validation set and camera frames, I could see I was just overfitting my training data; improvements on validation performance below 1% didn't reflect on improved detection in video frames.
   
Overall accuracy of the trained classifier on the validation set is 0.981. Here below a table with details of validation performance; Class 1 corresponds to cars, and Class 0 to no-cars.


| Class | Precision | Recall | F-score | Support |
|:-------:|:-----------:|:--------:|:---------:|:---------:|
|   0   |   0.984   | 0.988  |  0.986  |   2557  |
|   1   |   0.975   | 0.967  |  0.971  |   1199  |

When using validation to tune parameters, I tried to obtain a validation accuracy as high as possible, while maintaining precision and recall for both classes above 0.95 and, most important, solid (if subjective) detection of cars in video frames.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 What I found was that I could reach an accuracy on the validation set of up to 99% and still have detection miss cars in video frames, and give plenty of false positives. That was a sign of overfitting of the classifier, and perhaps an insufficient training dataset.

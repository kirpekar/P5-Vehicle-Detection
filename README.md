[//]: # (Image References)
[image1]: image-1.png
[image2]: image-2.png
[image3]: image-3.png
[image4-1]: image-4-1.png
[image4-2]: image-4-2.png
[image4-3]: image-4-3.png
[image4-4]: image-4-4.png
[image4-5]: image-4-5.png
[image5]: image5.png
[image6]: image6.png
[image7]: image7.png
[video1]: output_video.mp4
[test1]: test1.jpg

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

I am submitting three Jupyter notebooks -- 

01_Train_Save_Model.ipynb : This reads in all the training data, trains and model and saves it

02_Test_Static_Images.ipynb : This tests the model on the test static images

03_Test_Video.ipynb : This tests the model on the project video

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used the data provided by Udacity to train the model. As mentioned in the project framework - These example images came from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. 

![alt text][image1]

I used `scipy.misc.imread` to read in all files, so that took care of differences between JPEG, PNG, etc.

I used the function built in the classroom tutorial `get_hog_features` to extract hog features. This function calls `skimage.feature.hog`. This is built in the second code block of notebook-01.

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I explored all the color spaces available and the different parameters, but increasing the number of orientations & color spaces actually degraded performance during test (video). I think my linearSVC model is overfitting the data if I give it a larger number of features. So I eventually set it to:

```
color_space = 'RGB' 
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Use only the red channel
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I didn't do anything fancy really - scaled the data, split it into training/test sets and fit a linear SVC model.

```
#Scale all the data
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

#Split data for training/testing (80/20)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

#Fit Linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```

I then saved the scaler and model to a pickle file for future work.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I search only the blue box (Noting that the car is driving in the left most lane, I search only to the right).

![alt text][image3]

In this area, I started with 40x40px windows and scaled them up to 220x220px in increments of 20px. This ensured uniform coverage of all the area of interest, and at all scales.

![alt text][image4-1]
![alt text][image4-2]
![alt text][image4-3]
![alt text][image4-4]
![alt text][image4-5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

I used the red channel HOG, histogram on all three RGB channels and also 16x16 spatial binning.

I then ran these images through various search windows (documented above) and collected all the `hot_windows`. I then calculate the centroid of each hot window, as plotted by the red dots here


Original Image

![alt text][test1]


Centroids of positive matching windows

![alt text][image5]

I then send all the positive centroids to a clustering algorithm `sklearn.cluster.DBSCAN` which implements Density-based spatial clustering of applications with noise (DBSCAN). The original paper is available [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220)

The DBSCAN function returns the number of clusters (this requires some tuning of parameters to work correctly for this example) and the class of cluster that each input belongs to.

I then draw `n_clusters_` number of rectangles, centered at the centroid of all matches and with a width of 3-standard deviations in both x- and y- directions

![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My processed video is stored locally as `output_video.mp4`

YouTube Link here --

[![Alt Txt](http://img.youtube.com/vi/Ni-R8ZPFWKo/0.jpg)](http://www.youtube.com/watch?v=Ni-R8ZPFWKo)

https://www.youtube.com/watch?v=Ni-R8ZPFWKo


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For the video I used a slightly different pipeline copared to the static test images.

* Get frame from video
* Search for matching windows
* Create a heatmap initialized to zero, called `dark`
* For every matching window, increment `dark` by `+1` inside the matching window
* Threshold the heatmap, I found best results with `heat_threshold = 3`, i.e. choose only the pixels where at least 3 windows call out a match
* Send all matching pixels to DBSCAN for clustering
* Use output of DBSCAN to draw `n_clusters_` number of rectangles
* Draw rectangles such that they bound `(min-max)` the heatmap clusters detected

Here's an example result showing the heatmap from a video frame:

![alt text][image6]

Also, to ensure smooth frames and to reduce noise I blend the latest heatmap (5%) with the heatmap from the previous frame (95%). I then save the heatmap to use for the next frame. 

In other words I don't rely too much on a newly computed heatmap. I trust the accummulated heatmap information over time and only update it with a weight of 5%. This seems to work well - intuitively it will take a minimum of 20 frames to completely change a given heatmap, which corresponds to less than 1 second (video is 25fps).

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* For me the biggest problem was computational resources. It takes about 2.5 hours to render a 50 second video, so this is a long way off from real time application. 
* I found DBSCAN to be very reliable, as long as the `eps : The maximum distance between two samples for them to be considered as in the same neighborhood.` parameter was set correctly. I eventually set it to 50 pixels, because the approximate minimum detectable dimension of the car is roughly 100px and anything beyond that should be clustered as a different car/object.
* The pipeline was designed specifically for the project video and will fail in almost any other cicumstance, e.g. if the car changes lane, it will no detect anything to it's left, since that space is not being searched. 
* The pipeline can be made more robust by giving the training model a lot more data, by searching more extensively in the video frame and finally by building a more sophisticated vehicle tracking system. Currently I don't really track any cars, I simply identify them in every frame and smooth out the video, but one could implement a scheme to detect and then actively track a car while it is seen by the camera.

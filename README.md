
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visuals/chart.png "Visualization"
[image2]: ./visuals/bw_sign.png "Grayscaling"
[image3]: ./visuals/softmax_1.png "Softmax"
[image4]: ./web_images/web1.jpg "Traffic Sign 1"
[image5]: ./web_images/web2.jpg "Traffic Sign 2"
[image6]: ./web_images/web3.jpg "Traffic Sign 3"
[image7]: ./web_images/web4.jpg "Traffic Sign 4"
[image8]: ./web_images/web5.jpg "Traffic Sign 5"
[image9]: ./visuals/original_sign.png "Original Sign"
[image10]: ./visuals/fm_1.png "Feature Map 1"
[image11]: ./visuals/fm_2.png "Feature Map 2"
[image12]: ./visuals/fm_3.png "Feature Map 3"
[image13]: ./visuals/fm_4.png "Feature Map 4"
[image14]: ./visuals/fm_5.png "Feature Map 5"
[image15]: ./visuals/fm_6.png "Feature Map 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/GitDaBytes/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The bar chart below shows each of the 43 seperate traffic sign categories with the frequency of occurance of signs in that class. You can see that some signs have many examples, but many have far fewer. We can expect the underrepresented classes to be harder to learn without any furhter data augmentation.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the amount of data to use to train. This would likely reduce the number of weights needed in my net, speed up training and lower memory requirements and more. I also thought that color may be less important as cars would need to read signs at night where color information may not be available in low light. Also, it seemed like a lot of the images were very dark in the first place so I decided to try discarding color as a start.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image9]
![alt text][image2]

As a last step, I normalized the image data because I wanted to bring the data as close to possible to zero which greatly improves the training rate, and allows the weights to converge better. The data was normalized and centered around 0 (-1 to +1)

As additional steps to try and improve the net accuracy further, several steps could be taken / experimented with. Things that may help include:
* Generate new images for underrepresented classes in the data based on the existing data. Such generated images could be:
	* Rotated
	* Scaled
    * Flipped (where it makes sense to flip an image without changing meaning
    * Occluded
    * Placed on different backgrounds
* Images could undergo a number of processing steps such as:
  * Gaussian Blur
  * Brightness enhancement


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max Pooling			| 2x2 stride, outputs 5x5x16					|
| Dropout				| 0.5 keep rate									|
| Fully connected		| 200 Units										|
| RELU					|												|
| Dropout				| 0.5 keep rate									|
| Fully connected		| 84 Units										|
| RELU					|												|
| Fully connected		| 43 Units		
| Softmax				| 43 Classes        							|
|						|												|
|						|												|

The actual model is actually a modified LeNet model. I added dropout to the two fully connected layers, and also increased the number of units in the first fully connect layer. I found that increasing the number furhter did not improve accuracy. The dropout layers definitely made a big difference to the accuracy during training. I chose not to add dropout to the convolutional layers as this would increase training time, and by the way that they share weights across filters inherently provides a form of regularization.


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I experimented with a range of different hyperparameters. I ended up going with a learning rate of 0.001, a batch size of 64 and 50 epochs. The AdamOptimizer was used to reduce the mean cross entropy. I also built the model so I could change the dropout rate for dropout at any point in my model. I ended up selecting the standard 0.5 on the two layers with dropout as this seemed to yield the best results.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.4% 
* test set accuracy of 94.0%

In order to build my model, I decided to start with LeNet. I had read that LeNet is heavily used in handwriting recognition, such as on bank checks, and I had worked with it previously on the MNIST data set. The size of the sign images were similar to the MNIST image resolution, and both signs and the MNIST dataset have numbers in them, so I thought is worth a try.

After initially building the net, I trained it but could not get it higher than around 80 ~ 82%. After playing with hyperparameters, I contemplated augmenting the data set or tweaking the model. I decided to focus on the model. I played with filter levels in the CNN, but ultimately found that adding more units to the fully connected layers made a positive difference, as did adding dropout to those layers too. It was the dropout that made the biggest difference to performance. The dropout also seemed to take care of some overfitting I was seeing earlier too.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are all pretty clean and so I would expect to have a fairly good chance at predicting the signs. The last sign has some glare on the sign from the sign so this may make the task a little more difficult.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70km/h      			| 70km/h	   									| 
| Right of way 			| Right of way									|
| No entry				| No entry										|
| Yield	      			| Yield					 						|
| Road work				| Road work      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is a great start, but the images provided are clean. I would expect accuracy to drop if there were partial occlusions, large rotations, poor image quality, very low light and other factors that reduce clarity of the image.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 2st cell of the Ipython notebook.

Below I show five different predictions using the web based traffic signs I downloaded. For each prediction, The bar charts show the top five probabilities (per the net) as to what class the sign is in. As you can see, for each prediction, the net is extremely confident in its predictions. Thankfully they are correct predictions too. The probabilities shown (scale to 100) show were generated by a softmax layer on the network.

![alt text][image3]


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Below are images depicting a feature map from the first convolutional layer in the network. As is typical for a CNN, the earlier layers pick up very basic shapes. You can see from the images below that when we pass in the image of a yield sign, the filters in the first layer appear to be reacting strongly to lines.


![alt text][image10]

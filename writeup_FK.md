#**Traffic Sign Recognition - Build a Traffic Sign Recognition Project**


##The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/image_raw_neu.png "raw image"
[image2_1]: ./examples/image_gray_neu.png "image grayscaled"
[image2_2]: ./examples/image_standardized_neu.png "image standardized"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/sign1.png "Traffic Sign 1"
[image5]: ./examples/sign2.png "Traffic Sign 2"
[image6]: ./examples/sign3.png "Traffic Sign 3"
[image7]: ./examples/sign5.png "Traffic Sign 4"
[image8]: ./examples/sign6.png "Traffic Sign 5"
[image9]: ./examples/sign1_pred.png "Predicted Signs"
[image10]: ./examples/sign2_pred.png "Predicted Signs"
[image11]: ./examples/softmax.png "Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## 1. Writeup / Code

####1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MCFtm/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_FK.ipynb)

## 2. Data Set Summary & Exploration

####2.1 Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of validation set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2.2 Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th and 5th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per traffic-sign

![alt text][image1]

**ClassIds & SignNames as follows:**
0    Speed limit (20km/h);
1    Speed limit (30km/h);
2    Speed limit (50km/h);
3    Speed limit (60km/h);
4    Speed limit (70km/h);
5    Speed limit (80km/h);
6    End of speed limit (80km/h);
7    Speed limit (100km/h);
8    Speed limit (120km/h);
9    No passing;
10    No passing for vehicles over 3.5 metric tons;
11    Right-of-way at the next intersection;
12    Priority road;
13    Yield;
14    Stop;
15    No vehicles;
16    Vehicles over 3.5 metric tons prohibited;
17    No entry;
18    General caution;
19    Dangerous curve to the left;
20    Dangerous curve to the right;
21    Double curve;
22    Bumpy road;
23    Slippery road;
24    Road narrows on the right;
25    Road work;
26    Traffic signals;
27    Pedestrians;
28    Children crossing;
29    Bicycles crossing;
30    Beware of ice/snow;
31    Wild animals crossing;
32    End of all speed and passing limits;
33    Turn right ahead;
34    Turn left ahead;
35    Ahead only;
36    Go straight or right;
37    Go straight or left;
38    Keep right;
39    Keep left;
40    Roundabout mandatory;
41    End of no passing;
42    End of no passing by vehicles over 3.5 metric tons;

## 3. Design and Test a Model Architecture

####3.1 Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the seventh code cell of the IPython notebook.

I preprocessed the images in two steps:

In the first step I converted the images to grayscale. This might appear misleading as we have different colours in traffic signs. Thus a useful characteristic for sign recognition might have beeen erased. However I tried both ways training and testing the neural network with grayscaled and coloured images. The results were impressive. Trained with grayscaled images the CNN returned 100% correct recognition whereas it returned 75% trained on coloured images. 

In the second step I standardized (zero-centered & normalized). By this step noisy differences (e.g. in intensity) between images are egalized leading to a more efficient training of the CNN.

Here are examples of a traffic sign image before and after the steps of preprocessing.

raw image: 

![alt text][image2]

image after grayscaling:

![alt text][image2_1]

image after standardizing:

![alt text][image2_2]


####3.2 Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Image-data was already provided split into training-, validation- and test-data. the sizes of these portions were already presented in chapter 2.1. The ratio of having 12% validation-data of 100% training-data is sufficient considering the total amount of the data. Thus no further code for splitting the data into training and validation sets was applied, but stored as a backup at the end of the code.

####3.3 Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 12th cell of the ipython notebook. 

As a starting point for the model architecture The LeNet architecture was chosen, as recommended. Results were improved by implementing several preprocessing operations for the input-images. As this showed good results the general architecture wasn't changed. Here's the descriptions of the different layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image or 32x32x1 grayscale image	| 
| Convolution 5x5     	| Output = 28x28x6, 1x1 stride, padding: valid	|
| RELU					| Rectified Linear unit (ReLU)					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | Output = 10x10x16, 1x1 stride, padding: valid	|
| RELU					| Rectified Linear unit (ReLU)					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Flatten              	| input 5x5x6; output 400	          		    |
| Fully connected		| Input = 400; Output = 120  					|
| RELU					| Rectified Linear unit (ReLU)					|
| Fully connected		| Input = 120; Output = 84  					|
| RELU					| Rectified Linear unit (ReLU)					|
| Fully connected		| Input = 84; Output = 43    					|
|						|												|
 
####3.4 Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 14th and 16th cell of the ipython notebook. 

To train the model, I used the cross-entropy "cost function" which is calculated by the "tf.nn.softmax_cross_entropy_with_logits"-function. The logits, which are input to calculating the cross-entropy, are the result of running the input-data (images) through the neural network. Based on the cross-entropy the loss is calculated, which is then input to the optimizer-function that optimizes weights and biases of the neural network. As optimizer-function the AdamOptimizer was chosen. Adam is the abbreviation for Adaptive Moment Estimation (Adam). "This is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients, similar to momentum." (source: http://sebastianruder.com/optimizing-gradient-descent/index.html#adam)

####3.5 Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th cell of the Ipython notebook.

My final model results were:
* Training set accuracy s. Validation Accuracy = 0.980
* Test Accuracy = 0.892

If an iterative approach was chosen: No, as a starting point for the model architecture The LeNet architecture was chosen, as recommended. Results were improved by implementing several preprocessing operations for the input-images. As this showed good results the general architecture wasn't changed.

* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? As already described above as a starting point for the model architecture The LeNet architecture was chosen, as recommended. Results were improved by implementing several preprocessing operations for the input-images. As this showed good results the general architecture wasn't changed.
* Why did you believe it would be relevant to the traffic sign application? Actually I didn't have enough experience with CNNs to judge whether the LeNet would have the potential of solving the traffic sign recognition task. So I followed the recommendations of the instructors. However I noticed that it lead to good results without the need for major changes in architecture. Actually only the interfaces (input, output) between the different layers of the CNN had to be verified and adapated to the traffic sign application.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The model is working well, because its result on the validation set (0.98) exceeds the required result of 0.93 or greater.
 

## 4. Test a Model on New Images

####4.1 Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####4.2 Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 20th to 26th cell of the Ipython notebook.

Here are the results of the prediction (extract):

![alt text][image9] 

![alt text][image10] 


The model was able to correctly guess 16 of the 16 traffic signs, which gives an accuracy of 100%. 

####4.3 Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all images the model is astonishingly sure that this in detecting the correct signname. A significant 2nd ranked sign can only be seen for the "speed limit 50 km/h". The top five soft max probabilities were:

![alt text][image11] 

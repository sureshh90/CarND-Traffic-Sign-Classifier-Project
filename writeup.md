# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**


The goal of this project is to implement a Traffic Sign classifier.
The steps of this project are as follows:
* Load the data set
* Explore, summarize and visualize the data set
* Preprocess the dataset
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/class_distribution_training_set.jpg "Visualization"
[image2]: ./writeup_images/class_distribution_validation_set.jpg "Visualization"
[image3]: ./writeup_images/class_distribution_test_set.jpg "Visualization"

[image4]: ./writeup_images/training_1.jpg "Training"
[image5]: ./writeup_images/training_2.jpg "Training"
[image6]: ./writeup_images/training_3.jpg "Training"
[image7]: ./writeup_images/training_4.jpg "Training"
[image8]: ./writeup_images/training_5.jpg "Training"
[image9]: ./writeup_images/training_6.jpg "Training"

[image10]: ./new_test_images/01-Speed-limit-30-km-h.jpg "Images from internet"
[image11]: ./new_test_images/11-Right-of-way-at-the-next-intersection.jpg "Images from internet"
[image12]: ./new_test_images/12-Priority-road.jpg "Images from internet"
[image13]: ./new_test_images/14-Stop.jpg "Images from internet"
[image14]: ./new_test_images/15-No-vehicles.jpg "Images from internet"


[image15]: ./writeup_images/softmax_prob_01-Speed-limit-30-km-h.jpg "Softmax Probabilities"
[image16]: ./writeup_images/softmax_prob_11-Right-of-way-at-the-next-intersection.jpg "Softmax Probabilities"
[image17]: ./writeup_images/softmax_prob_12-Priority-road.jpg "Softmax Probabilities"
[image18]: ./writeup_images/softmax_prob_14-Stop.jpg "Softmax Probabilities"
[image19]: ./writeup_images/softmax_prob_15-No-vehicles.jpg "Softmax Probabilities"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. 

Numpy library is used to calculate the statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
    (Here the shape represent the size of the scaled image that is available in the ```test['features']``` array. It does not represent the size of the original image.)
* The number of unique classes/labels in the data set is **43**

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

A bar chart representing the number of examples for each class has been plotted.

### Training Set

![alt text][image1]

### Validation Set

![alt text][image2]

### Test Set

![alt text][image3]

A quick view of the above bar graphs reveal that the distribution of the images/data across different classes is uneven. It can be seen that some classes have more images than others. 

A quick judgement from the bar graphs is that the classifier might perform better to identify classes in the first half (1-20) of class indices than the second half (21-43). The reason being that the network, in the second half, has fewer images to train on.

### Design and Test a Model Architecture

#### 1. Preprocessing the image data. 

As a first step, I decided to shuffle the images since the order of images should not play a role in the training of the network.


As the next step, two methods for normalising data were considered. The aim of normalisation is to limit the input data (i.e. the pixel values) to a small range. This helps the loss reduction algorithm to converge faster and prevents the algorithm to get stuck on a local minima.


#### 1.1 Simple Normalisation

Here the normalisation technique used is ```(pixel - 128)/ 128```.

After applying this normalisation technique, the pixels values were limited to the range 0 - 1.992 for the training set.


#### 1.2 MinMax Scaling

Here the normalisation technique used is ```a + ( ( (image_data - scale_min)*(b - a) )/( scale_max - scale_min ) )```

After applying this normalisation technique, the pixels values were limited to the range 0.1 - 0.9 for the training set.

Among the two normalisation techniques, the second one (MinMax Scaling) is chosen as it helped us in reaching the desired validation accuracy.



#### 2. The model architecture including model type, layers, layer sizes, connectivity, etc.

I used the LeNet architecture described in the lectures as a base architecture [3]. Some dropout layers were added to the architecture to improve the validation accuracy. The reasons would be discussed in the next section.


The final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, Output 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, Output 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Output 5x5x16    				|
| Flatten       		| Output 400 									|
| Fully Connected		| Input 400 Outputs 120							|
| RELU					|												|
| Dropout				|												|
| Fully Connected		| Input 120 Outputs 84							|
| RELU					|												|
| Dropout				|												|
| Fully Connected		| Input 84 Outputs 43							|
| 						|												|
 


#### 3. Training the model. 

To train the model, AdamOptimizer is used. 

AdamOptimizer is different from **Stochastic Gradient Descent** in that SGD maintains a single learning rate and the learning rate does not change during training. [1]

AdamOptimizer uses the best of two worlds of the following two algorithms.

* **Adaptive Gradient Algorithm (AdaGrad)** maintains a per-parameter learning rate. This improves performance on problems with sparse gradients like natural language and computer vision problems. [2]

* **Root Mean Square Propagation (RMSProp)** that also maintains per-parameter learning rates. These rates are adapted based on the average of recent magnitudes of the gradients for the weight (i.e. based on rate of change). [2]


#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

The final model yielded the following results:

* Training Set Accuracy = **0.958**
* Validation Set Accuracy = **0.953**
* Test Set Accuracy = **0.932**

The code can found in the section '**Train the model**' section.

An iterative approach was applied to reach a validation set accuracy of atleast **0.93**.

* The initial architecture chosen was the LeNet architecture that was explained in the lecture.

* The performance of the architecture was first analysed. As seen from the below figure, we could reach an accuracy of around **0.9**. It could also be seen that the training accuracy was nearing **1**, which implies the network is overfitting.
The learning rate is **0.001**, the batch size is **128** and only shuffling is done as preprocessing step.

   ![alt text][image4]


* In order to reduce, the overfitting from the previous step, dropout layers were introduced. Because of the introduction of the dropout layers (with the keep probability of **0.5**), the number of epochs were not enough to completely reduce the loss. This resulted in a validation accuracy of **0.7**. It can also be noted that the training accuracy is **0.62**, which means the network is underfitting. The other parametes are same as the previous step.

    ![alt text][image5]

* In order to reduce the underfitting from the previous step, one of the following methods could be applied. 
        * Increase the number of epochs to reduce the loss.
        * Increase the keep probability so that the learnt representations are retained. 
          (Caution should be applied not to increase it much since it leads to overfitting.)
  
  The below figure shows the accuracy with the parameters number of epochs as **100** and keep probability of **0.6**. It could be noticed that validation and training accuracy are close to each other (which indicates a good training pattern) and validation accuracy is around **0.95**. The other parametes are same as the previous step.
     
     ![alt text][image6]
     
* In the above step, even though the validation accuracy is as expected, the number of epochs to train them is huge, which in turn means that the algorithms takes many steps to converge to the minimum. 

    One reason for this behaviour can be attributed to the preprocessing step. Since the input data (pixel) has a large range, it could be difficult for the algorithm to converge. Therefore, as the next step, normalisation techniques were added to the preprocessing step. The other parametes are same as the previous step.
    
    A few trails with Simple Normalisation (**Section 1.1**) revealed that it did not increase the validation accuracy much.
    
    An example figure is shown below.
    
    ![alt text][image7]
    
     Errata: In the image the preprocessing step is shown as Only Shuffle but it should be Shuffle + Easy Normalisation.
    
    Hence the MinMax Scaling was used as normalisation technique and with keep probability of **0.4** we could achieve a validation accuracy of around **0.80**. The other parametes are same as the previous step. 

    ![alt text][image8]
   
    Errata: In the image the preprocessing step is shown as Only Shuffle but it should be Shuffle + MinMax Scaling.
   
* As the final step, to increase the training accuracy (or to reduce underfitting) the keep probability was increased to **0.5**. The number of epochs has been reduced to **35** since the accuracy started to saturate and then decrease. (This calibration is similar to early termination but done manually). The other parametes are same as the previous step. 
    
    ![alt text][image9]
    
    Errata: In the image the preprocessing step is shown as Only Shuffle but it should be Shuffle + MinMax Scaling.
   

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]

The images comes in higher resolution and have to scaled down to **32 * 32 * 3** image. This results in loss of quality of the input image. The output images from resizing operation seems to be more pixelated which might result in bad classifications.


The first image might be difficult to classify because the pixelation on the image is large. Since usually numbers are difficult to classify and in addition this pixelation increases the difficulty of classification.

The other images eventhough pixelated, the features are still intact and less likely to be confused with other traffic signs.

#### 2. Discussion on the model's predictions on these new traffic signs and comparison of the results to predictinon on the test set. 

Here are the results of the prediction:

| Image									    |     Prediction			   					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| 30 km/h	      							| 30 km/h    					 				|
| Right of way at the next intersection		| Right of way at the next intersection 		|
| Priority Road								| Priority Road									|
| Stop Sign      							| Stop sign   									| 
| No vehicles 								| No vehicles									|


The model was able to correctly guess 5 out of the 5 traffic signs, which gives an accuracy of **100%**. 

Comparing to the accuracy on the test set **(94%)** this is high. The reason could also be attributed to the number of examples in the new test set (which is currently five).


#### 3. Discussion on how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

For the first image, the model is relatively sure that the sign is **30 kmph sign** and the image is **30 kmph**. The top five soft max probabilities are plotted in a bar graph below.

![alt text][image15]


**
For the second image, the model is sure (with probability of 1) that the sign is **Right of way at the next intersection**, and the image does contain a that sign. The top five soft max probabilities are plotted in a bar graph below.

![alt text][image16]



For the third image, the model is sure (with probability of 1) that the sign is **Priority Road**, and the image does contain a that sign. The top five soft max probabilities are plotted in a bar graph below.

![alt text][image17]


For the fourth image, the model is sure (with probability of 1) that the sign is **Stop**, and the image does contain a that sign. The top five soft max probabilities are plotted in a bar graph below.

![alt text][image18]


For the fifth image, the model is completely not sure about the **No Vehicle** sign, but classifies the sign with a probability of around **0.6**. The next closest prediction is **50 kmph**. The top five soft max probabilities are plotted in a bar graph below.

![alt text][image19]

### Further Improvements

The project could be extended by adding the following features.
* The architecture could be improved further by adding new layers.
* Early termination feature for preventing overfitting could be added.
* The distribution of input classes is uneven. In order to compensate for this, data augumentation techniques could be used.
* The precision and recall values for the various classes in the test set can be calculated and can be used for data augumentation and fine tuning the model.
* The layers of the neural networks could be visualised to understand more about the model.

### References:

* [1] The lecture material on Stochastic Gradient Descent.
* [2] Link: [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).
* [3] The code used in this project is heavily based on the code available from exercises/assignments from the lectures.



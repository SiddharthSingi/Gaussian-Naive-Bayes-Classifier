# Gaussian-Naive-Bayes-Classifier
Implemented my own GaussianNB Classifier using Python. The classifier uses data of positions and velocities of cars and predicts whether the car will continue straight or take a left or right turn.


To Understand Gaussian Navie Bayes, first lets go through some probabilistic math:


Unfortunately I cannot easily show math equations in a Markdown file so please have a look at the following images to understand how a Gaussian Naive Bayes Classifier works:


![img_20180312_173550](https://user-images.githubusercontent.com/26694585/37283024-7ddc55ce-261c-11e8-91be-d6b9f127b717.jpg)

![img_20180312_173626](https://user-images.githubusercontent.com/26694585/37283025-7e13f632-261c-11e8-844f-3e2864616afe.jpg)


For more information on how Naive Bayes Classifiers work have a look: 
[Naive Bayes Classifier Explained](https://appliedmachinelearning.wordpress.com/2017/05/23/understanding-naive-bayes-classifier-from-scratch-python-code/)


In the train.json file for every data point I have 4 features of the car, they are:
* s (position of the car along the length of the road)
* s_dot (velocity of the car along the length of the road)
* d (distance of the car along the perpendicular direction of the road)
* d_dot (velocity of the car along the perpendicular direction of the road.)

The labels of the car may be either of "left", "keep", or "right".


## Working
I have created a train method in classifier.py which computes the means and standard deviations of every feature, of every label for all the data points. These values are used to predict direction a car will go into using only the 4 features of the car. Whenever the classifier sees a new set of features in the test.json file it uses the previously calculated values to find out the Gausian Probability of whether the car is about to go straight or take a lefft or right turn. This is implemented in the prediction.py file.

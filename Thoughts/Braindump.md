#Braindump

###Overview

This is my braindump of info about using neural networking for image processing.

The techniques used fall under one of 2 categories:

1. Improving the training accuracy
2. Generalising to improve the validation accuracy

The latter will generally come at the cost of the former, but since the end result is normally to have a network which can work well with previously unseen data, the validation accuracy (and/or test accuracy) is the one we need to improve. Improving the training accuracy is only useful insofar as improving the validation accuracy.

###Improving training accuracy

These are some of the techniques:
- Convolutional layers
- Cross entropy
- Different optimisers

###Generalising to improve validation data

These are some of the techniques:
- Data augmentation
- Dropout
- L1 and L2 regularisation

###Convolutional layers

These allow the network to apply filters to the image, generally allowing it to change the contrast of it. This, in turn, can make various edges clearer and better defined, which can make the image easier to classify.

###Cross entropy

This is a way of using as much of the training data as possible to make an estimate of the accuracy.

###Different optimisers

Gradient descent is only the simplest optimisers. There are also more complicated optimisers (e.g. Adam, Adagrad) that keep track of how many times each label has been seen, and will make the network have a bigger change when rarer examples are found.

###Data augmentation

This is used to artificially add training data, and is done by tweaking the existing training images so that they are still recognisable, but have a different input. This requires some knowledge of what is 'acceptable' tweaking.

###Dropout

You set some weights to zero randomly. This forces the network to classify the inputs without full use of internal nodes, which makes the network more resilient and robust.

###L1 and L2 regularisation

These basically add an amount to the cost, which penalises the network for having large weights.

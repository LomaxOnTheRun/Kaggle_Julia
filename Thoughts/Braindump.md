#Braindump

###Overview

This is my braindump of info about using neural networking for image processing.

The techniques used fall under one of 2 categories:

1. Improving the training accuracy
2. Generalising to improve the validation accuracy

The latter will generally come at the cost of the former, but since the end result is normally to have a network which can work well with previously unseen data, the validation accuracy (and/or test accuracy) is the one we need to improve. Improving the training accuracy is only useful insofar as improving the validation accuracy.

###Improving training accuracy

These are some of the techniques:
- Multiple layers
- Convolutional layers
- Cross entropy
- Different optimisers

###Generalising to improve validation data

These are some of the techniques:
- Data augmentation
- Dropout
- L1 and L2 regularisation

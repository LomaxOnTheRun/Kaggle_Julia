#Building our first neural network

###Overview

In this post, we'll be creating our first and most basic neural network, using TensorFlow. This is Google's contribution to the field of open-sourced neural network libraries. It uses a C++ backend, which allows for faster computations, although it can make it trickier to use to begin with, and more difficult to implement and tweak your own training algorithms later on. However, for our purposes it will do very well.

I should mention now that there is a lot going on in this script. As with the last post, I'll go through and explain each part as I go, but there are quite a few aspect of this that we need to look at to create a coherent and useful script. There is a link at the bottom to the finished script, so feel free to play around with it first, then come back and read up on any parts that don't make sense.

This script will be roughly broken down into 4 parts:

1. Loading the pickled data
2. Setting up the network
3. Running the network
4. Plotting the network accuracy

So without further ado:

###Loading the pickled data

We first of all need to get our data from the pickle file that we created in the [last post] (/master/blog/Blog_1__Getting_the_training_data.md).

```python
import pickle
import numpy as np

# Gets the training data and labels out of the pickle file
with open('julia.pickle', 'rb') as f:
	save = pickle.load(f)
	trainDataset = save['trainDataset']
	trainLabels = save['trainLabels']
	validDataset = save['validDataset']
	validLabels = save['validLabels']
	testDataset = save['testDataset']
	testLabels = save['testLabels']
	del save  # Hint to help garbage collection free up memory
```

As I mentioned ast time, sanity checks are a wonderful thing. An easy one to check we've loaded the correct datasets is to have a look at the shapes of the datasets and labels. 'Shape' in this case refers to the sizes of the n-dimentional matrices we're using.

```python
def checkShapes():
	print 'Training set:', trainDataset.shape, trainLabels.shape
	print 'Validation set:', validDataset.shape, validLabels.shape
	print 'Test set:', testDataset.shape, testLabels.shape
```

If you run the above code, you should get an output that looks like this:

```
>>> Training set: (4283, 20, 20) (4283,)
>>> Validation set: (1000, 20, 20) (1000,)
>>> Test set: (1000, 20, 20) (1000,)
```

This looks correct, as we decided to put 1000 images aside for validation and testing, and each of the images are 20x20 pixels. The labels are just 1D vectors of IDs.

###Refactoring the code

Now, TensorFlow actually requires 

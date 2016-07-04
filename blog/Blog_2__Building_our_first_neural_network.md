#Building our first neural network

###Overview

In this post, we'll be creating our first and most basic neural network, using TensorFlow. This is Google's contribution to the field of open-sourced neural network libraries. It uses a C++ backend, which allows for faster computations, although it can make it trickier to use to begin with, and more difficult to implement and tweak your own training algorithms later on. However, for our purposes it will do very well.

I should mention now that there is a lot going on in this script. As with the last post, I'll go through and explain each part as I go, but there are quite a few aspect of this that we need to look at to create a coherent and useful script. There is a link at the bottom to the finished script, so feel free to play around with it first, then come back and read up on any parts that don't make sense. The script will be roughly broken down into 4 parts:

1. Loading the pickled data
2. Setting up the network
3. Running the network
4. Plotting the network accuracy

###Network structure

For our initial foray into neural networks, we'll stick with a very simple structure of 3 layers:

- **The input layer.** This is simply how we will 'show' the network our images. In this case, it will be a layer 400 nodes wide, one for each pixels in the image.
- **The hidden layer.** This is a layer of nodes which we don't interact with. Every node in the hidden layer is connected to every node in the input layer, and every one of these connections has a weight associated to it. Each node in the hidden layer also has a bias applied to it, and it is by adjusting these weights and biases that the network 'learns'.
- **The output layer.** This is the layer from which we get the networks's assesment of the picture. There are as many output nodes as label IDs, and the output of each node shows how strongly the network believes the image we showed it has a particular ID. So the higher the output of node 10 is, the more strongly the network believes the image is of an '**A**'. Every output node is connected to every hidden node, and also has adjustable weights and a bias.

###Loading the pickled data

We first of all need to get our data from the pickle file that we created in the [last post] (/blog/Blog_1__Getting_the_training_data.md).

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

As I mentioned last time, sanity checks are a wonderful thing. An easy check here is to look at the shape of the datasets and labels we've just loaded. 'Shape' in this case refers to the sizes of the n-dimentional matrices we're using.

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

###Refactoring our datasets and labels

Whilst our network layer has an input node for each pixel in the image, it doesn't actually have a 2D structure like our images have. This means that we need to reshape our each image so that the pixels are represented by a single vector, rather than a 2D matrix.

At the same time, our labels also need to be reformatted. This is because we want them to look like the 'ideal' network output. This 'ideal' output is a vector, 62 values long, made up entirely of zeros with the exception of the target output, which is a one. So if the image is of a '**2**', then the output vector will be [0, 0, 1, 0, ..., 0, 0], where all the values not shown are zeros.

We do both of these reformats in a single function, then apply it to all datasets and labels.

```python
def reformat(dataset, labels):
	'''Reformats datasets and labels'''
	dataset = dataset.reshape((-1, dataset.shape[1] * dataset.shape[2])).astype(np.float32)
	labels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# Fixed variables
imageSize = 20
numLabels = 62

# Reformat the data
trainDataset, trainLabels = reformat(trainDataset, trainLabels)
validDataset, validLabels = reformat(validDataset, validLabels)
testDataset, testLabels = reformat(testDataset, testLabels)
```

We can now do another sanity check by running ```checkShapes()``` again, this time after the reformat. We should be able to see that both the datasets and the labels are 2D matrices.

```
>>> Training set: (4283, 400) (4283, 62)
>>> Validation set: (1000, 400) (1000, 62)
>>> Test set: (1000, 400) (1000, 62)
```

###Creating the network

We're now going to set up the network, using TensorFlow objects. I'll try to explain everything as I go, but if you're still not quite sure what's going on, I recommend going to TensorFlow's [website] (https://www.tensorflow.org/versions/r0.9/tutorials/index.html), which goes over how to build networks for learning MNIST data. I have found these enormously helpful whilst trying to get a handle on using TensorFlow. Anyway, on with the show.

The first thing we're going to do is to create placeholders for our batch data. When we train our network, we're going to show it a number of images before letting it backpropagate the error and adjust the weights. Using a placeholder means that a space in memory is allocated to this variable, while allowing us to change its value once the session has started. In this case, we're going to fill these placeholders with sets of images we want to network to learn from.

```python
def trainNeuralNetwork(plots):
	'''Run the TensorFlow network'''

	# Placeholders for batch data
	tfBatchDataset = tf.placeholder(tf.float32, shape=(batchSize, imageSize * imageSize))
	tfBatchLabels = tf.placeholder(tf.float32, shape=(batchSize, numLabels))
```


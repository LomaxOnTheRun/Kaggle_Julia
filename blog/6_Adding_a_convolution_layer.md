# Adding a convolution layer

### Overview

We're going to be adding a convolution layer to our network now. This is going to allow the network to create several 'filters' which it will apply to each of the images, and it will be the filtered images which will be put through the 2 hidden layers. Convolution layers are substantially lighter than their hidden layer counterparts, as only the kernels get updated, not every connection weight for every node.

### Adding the layer

As before, the first thing we need to change it the creation of our weights and biases, this time to include the convolution layer. There's 3 extra hyperparameters we need to be aware of:

 - *patchSize*. This is the length of each of the sides of the patch we'll be using.
 - *numChannels*. When using a convolution layer, it assumes that it will be looking at a stack of images. Since our images are greyscale, this will be set to 1. We'll also need to reformat our data, but more on that later.
 - *depth*. This is the number of different filters that the convolution layer will create and apply to the input images. The higher this number, the more ways the network can look at the image.

```python
def getWeightsAndBiases():
	'''Create initial weights and biases'''
	# Weights
	w_conv1 = tf.Variable(tf.truncated_normal([patchSize, patchSize, numChannels, depth], stddev=0.1))
	w_hidden1 = tf.Variable(tf.truncated_normal([imageSize * imageSize * depth, numHidden], stddev=0.1))
	w_hidden2 = tf.Variable(tf.truncated_normal([numHidden, numHidden], stddev=0.1))
	w_output = tf.Variable(tf.truncated_normal([numHidden, numLabels], stddev=0.1))
	tfWeights = {'conv1' : w_conv1, 'hidden1' : w_hidden1, 'hidden2' : w_hidden2, 'output' : w_output}
	# Biases
	b_conv1 = tf.Variable(tf.zeros([depth]))
	b_hidden1 = tf.Variable(tf.zeros([numHidden]))
	b_hidden2 = tf.Variable(tf.zeros([numHidden]))
	b_output = tf.Variable(tf.zeros([numLabels]))
	tfBiases = {'conv1' : b_conv1, 'hidden1' : b_hidden1, 'hidden2' : b_hidden2, 'output' : b_output}
	return tfWeights, tfBiases
```

The next thing we need to change is our logits logic. We need to add the ```tf.nn.conv2d()``` layer, then pass it through a ReLU filter, then reshape the matrix so that we can pass it to our first hidden layer.

```python
def getLogits(tfBatchDataset, tfWeights, tfBiases, dropout=1.0):
	'''Runs images through the network and returns logits'''
	tfConv1 = tf.nn.relu(tf.nn.conv2d(tfBatchDataset, tfWeights['conv1'], [1, 1, 1, 1], padding='SAME') + tfBiases['conv1'])
	tfShape = tfConv1.get_shape().as_list()
	tfReshape = tf.reshape(tfConv1, [tfShape[0], tfShape[1] * tfShape[2] * tfShape[3]])
	tfHidden1 = tf.nn.relu(tf.matmul(tfReshape, tfWeights['hidden1']) + tfBiases['hidden1'])
	tfHidden2 = tf.nn.relu(tf.matmul(tfHidden1, tfWeights['hidden2']) + tfBiases['hidden2'])
	tfLogits = tf.matmul(tfHidden2, tfWeights['output']) + tfBiases['output']
	return tfLogits
```

We also need to change how we reformat our datasets at the start of our script. Since we need to accomodate the convolution layer looking for a stack of images, rather than a single image at a time, we'll need to place each pixel in a mini stack of size 1.

```python
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, imageSize, imageSize, numChannels)).astype(np.float32)
	labels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return dataset, labels
```

The final two bits of logic we need to change are how we get the batch data to our network. This is because the shape of our matrices have changed, and so we need to both create the ```tfBatchDataset``` placeholder and then fill it accordingly.

```python
	tfBatchDataset = tf.placeholder(tf.float32, shape=(batchSize, imageSize, imageSize, numChannels))
```

```python
def getBatchData(step):
	'''Get one of each type of image'''
	# Images
	batchImages = np.ndarray(shape=(62, imageSize, imageSize, numChannels), dtype=np.float32)
	for index, imageBin in enumerate(imageBins[:62]):
		showStep = step
		while showStep >= len(imageBin):
			showStep -= len(imageBin)
		image = imageBin[showStep]
		batchImages[index, :, :, :] = image
	# Labels
	labels = np.arange(numLabels)
	batchLabels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return batchImages, batchLabels
```

We're now going to try to run our network with some initial values for our new hyperparameters. These values were many gotten by looking at other people's networks to see what values are used in the commnity at large.

```python
numChannels = 1
patch_size = 5
depth = 16
```

We now get a graph that looks something like this:

![Graph 1](/images/Julia_6_blog_1.png)

As we can see, the accuracy becomes very good very quickly. However, what the graph doesn't show is that each step now takes significantly longer. No problem there, we can just reduce the amount of steps to something more managable, and increase the frequency of our checks.

```python
numSteps = 2001
progressCheckInterval = 50
```

This now gives us a more informative graph that doesn't take an age for the script to produce.

![Graph 2](/images/Julia_6_blog_2.png)

By adding in the convolution layer, we've managed to increase the accuracy by 5~10% from our previous network. This is great, although the accuracy fluctuates a lot, even after several thousend steps. We'll look at how to reduce this in the next post.

### The full script

You can get the full script [here](/blog/Julia_6.py).

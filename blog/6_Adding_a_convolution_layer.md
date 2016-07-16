#Adding a convolution layer

###Overview

We're going to be adding a convolution layer to our network now. This is going to allow the network to create several 'filters' which it will apply to each of the images, and it will be the filtered images which will be put through the 2 hidden layers. Convolution layers are substantially lighter than their hidden layer counterparts, as only the kernels get updated, not every connection weight for every node.

###Adding the layer

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

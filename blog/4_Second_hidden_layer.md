#Adding a second hidden layer

###Overview

We're now going to add a second hidden layer to our neural network, and the problems that arise from doing so.

###Adding the layer

To start with, we're going to keep the same number of nodes in each of our hidden layers, to help keep down the number of hyperparameters to test. We then need to change our code that creates the weights and biases.

```python
def getWeightsAndBiases():
	'''Create initial weights and biases'''
	# Weights
	w_hidden1 = tf.Variable(tf.truncated_normal([imageSize * imageSize, numHidden]))
	w_hidden2 = tf.Variable(tf.truncated_normal([numHidden, numHidden]))
	w_output = tf.Variable(tf.truncated_normal([numHidden, numLabels]))
	tfWeights = {'hidden1' : w_hidden1, 'hidden2' : w_hidden2, 'output' : w_output}
	# Biases
	b_hidden1 = tf.Variable(tf.zeros([numHidden]))
	b_hidden2 = tf.Variable(tf.zeros([numHidden]))
	b_output = tf.Variable(tf.zeros([numLabels]))
	tfBiases = {'hidden1' : b_hidden1, 'hidden2' : b_hidden2, 'output' : b_output}
	return tfWeights, tfBiases
```

We then need to change how we calculate our logits, to include our new layer.

```python
def getLogits(tfBatchDataset, tfWeights, tfBiases, dropout=1.0):
	'''Runs images through the network and returns logits'''
	tfHidden1 = tf.nn.relu(tf.matmul(tfBatchDataset, tfWeights['hidden1']) + tfBiases['hidden1'])
	tfHidden2 = tf.nn.relu(tf.matmul(tfHidden1, tfWeights['hidden2']) + tfBiases['hidden2'])
	tfLogits = tf.matmul(tfHidden2, tfWeights['output']) + tfBiases['output']
	return tfLogits
```


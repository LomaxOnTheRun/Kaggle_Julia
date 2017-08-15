# Adding a second hidden layer

### Overview

We're now going to add a second hidden layer to our neural network, and the problems that arise from doing so.

### Adding the layer

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

Thankfully the code that calculates the L2 regularisation already takes into account all the weights and biases we create and put into our ```tfWeights``` and ```tfBiases``` dictionaries, so we don't need to change anything there. When we now run our code, we should get something that resembles this:

![Second hidden layer 1](/images/Julia_4_blog_1.png)

Huh, that doesn't look like our normal graph. It looks like the network gets stuck for a few thousend steps, before eventually sorting itself out. In fact, if you run this a few times, you should see that it's getting stuck at the same plce every time: 7.4% for the training data, and 6.2% for the validation data. This is suspicious, as the network normally still jups around a little when we've maxed out our accuracy. If we extend the amount of steps we allow our network to 3000, we get something that looks a lot more like what we're used to seeing.

![Second hidden layer 2](/images/Julia_4_blog_2.png)

We can see that our network learns in a very similar way to normal, once it's broken through the plateau. So what's getting it stuck?

### Investigating the plateau

We can change our code slightly to get a better idea of what's happening in our network. We're going to make the network show us every correct guess it makes.

```python
def accuracy(predictions, labels, showGuesses=False):
	'''Check if most likely network outcomes are correct'''
	if showGuesses:
		guesses = []
		for index, prediction in enumerate(predictions):
			guess = np.argmax(prediction)
			if (guess == np.argmax(labels[index])):
				guesses.append(guess)
		print guesses
		return (100.0 * len(guesses)) / predictions.shape[0]
	else:
		maxPredictions = np.argmax(predictions, 1)
		maxLabels = np.argmax(labels, 1)
		numCorrectPredictions = np.sum(maxPredictions == maxLabels)
		return (100.0 * numCorrectPredictions / predictions.shape[0])
```

We then just need to set that ```showGuesses``` flag to true in our runs, to see what answers it's getting correct.

```python
				trainAccuracy = accuracy(tfTrainPrediction.eval(), trainLabels, showGuesses=True)
```

If we run the code now, we can see that the network very quickly gets stuck exclusively getting '10's correctly, which correspond to 'A's. Let's take a look at the breakdown of our validation dataset, to see if it can throw some light on this. Note that we're using the validation dataset instead of the training dataset as it's smaller and therefore generally faster to run diagnotics on. We put the follow code in ust before we reformat our datasets and labels.

```python
def showLabelCounts(labels):
	for index in xrange(62):
		count = (labels == index).sum()
		print 'ID: %s\t\tCount: %s' % (index, count)

showLabelCounts(validLabels)
```

So it looks like there are a lot more 'A's (or '10's) in our validation dataset than any other label. In fact, there are 62 '10's, which matches up exactly with the accuracy of our network during the plateau (remember, we have 1000 data points in our validation dataset). It's reasonable to guess that the network is probably just using a very simplistic strategy of guessing the most common label for every input. We can check if our theory holds for the training data, by showing the training label counts.

```python
showLabelCounts(trainLabels)
```

There are 315 counts out of 4283 total trainining data points, which gives us a percentage of 7.35, which matches our plateau value. This means our network has come to a saddle point, which only [gets worse the more complex our network gets](https://arxiv.org/abs/1406.2572). There are a few ways of minimising this problem (changing learning rates, momentum, etc.), but our network will hit a point where we'll need to deal with it directly, or it will get stuck on this plateau permanently. We'll see how to do this in the next post.

###Full code

The full script for this post can be found [here](/blog/Julia_4.py).

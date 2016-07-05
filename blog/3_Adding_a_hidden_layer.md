#Adding a hidden layer

###Overview

In this post, we'll add a hidden layer to our network, improving its capabilites significantly. We'll also look at how to pick hyperparameters with new networks like this.

###The weights and biases

The first change we're going to have to make to our code is to add a matrix and a vector for our new sets of weights and biases respectively. To do this, we simply need to change our ```getWeightsAndBiases()``` function to return several weight matrices and bias vectors. We're going to do this by using dictionaries. This allows us to store them in an efficient way, and then to be able to refer to specific weight and bias matrices by name, making it nicer to read.

```python
def getWeightsAndBiases():
	'''Create initial weights and biases'''
	# Weights
	w_hidden = tf.Variable(tf.truncated_normal([imageSize * imageSize, numHidden]))
	w_output = tf.Variable(tf.truncated_normal([numHidden, numLabels]))
	tfWeights = {'hidden' : w_hidden, 'output' : w_output}
	# Biases
	b_hidden = tf.Variable(tf.zeros([numHidden]))
	b_output = tf.Variable(tf.zeros([numLabels]))
	tfBiases = {'hidden' : b_hidden, 'output' : b_output}
	return tfWeights, tfBiases
```

###The logits

The other function we need to change is how the logits are calculated. We do this by calculating the 'logits' for the hidden layer, then applying a RELU function to them, then calculating the logits for the output layer from the hidden layer. The RELU function acts to non-linearise the hidden logits. There are several ways of doing this, but RELUs are very cheap computationally.

```python
def getLogits(tfBatchDataset, tfWeights, tfBiases):
	'''Runs images through the network and returns logits'''
	tfHidden = tf.nn.relu(tf.matmul(tfBatchDataset, tfWeights['hidden']) + tfBiases['hidden'])
	tfLogits = tf.matmul(tfHidden, tfWeights['output']) + tfBiases['output']
	return tfLogits
```

###Bits and bobs

Don't forget to change your graph label from 'Julia 1' to 'Julia 2'.

###Number of hidden layers

The sharper eyed readers may have noticed that we snuck in an extra variable, ```numHidden```, in the functions defined above. This is an extra hyperparameter for our network, and is just the amount of nodes we want our hidden layer to have. But how do we choose a new hyperparameter?

The first thing to do is to guess a rough ballpark figure. We can guess that if we have 1 node, we're going to cripple our network, as all the output nodes will just have a single connection going to each one. If we have a 1,000,000 nodes, our network is going to run very slowly, and since we're still just playing with rough numbers, we want to very quickly get an idea if our choice of hyperparameter is rubbish. So, since we have 400 input nodes and 62 output nodes, we might think that 100 nodes might be roughly correct.

We're now going to run our program 3 times, once with our guess, once with a value 10x smaller than our guess, and once with a value 10x larger than our guess. This way we should have a much better idea of what's going to work for our network.

![Comparison 1](/images/Julia_2_blog_1.png)

As we can see from the graph, the more nodes we have, the better the training and validation results become, but the more overfitted the network becomes. We'll look at ways to decrease overfitting later on (and so raise the validation results closer to the training results), but a network that's overfitting too much is probably too complex for the task and is likely to take longer than an ideal network.

The other key issue to consider here is the time taken for all of these to run. The network with 10 hidden nodes took about 60s to run, the network with 100 hidden nodes took about 100s to run, and the network with 1000 nodes took about 1400s (roughly 23 minutes) to run. Having our still very basic network take 20 minutes each time we want to run it is far too long, so that one's out. There isn't enough of a difference in time between the other two for it to be a factor, so we'll go with the more successful 100 node network.

###Number of steps

In the graph above, we can clearly see the validation accuracy leveling out while the training accuracy keeps getting better. Since we don't actually care about the training validation, beyond using it to see how much overfitting is happening, we want to only train our network for as long as it takes for the validation accuracy to hit it's maximum and level out. Going by the graph, it looks like the validation accuracy completely stopped improving after about 7000 steps. If we wanted to, we could choose to run our network for only 7000 steps, to save us the extra time. However, since *the extra time* is only going to be a few seconds, we can keep the current number of steps as it lets us see if something we change helps it to continue learning after 10,000 steps.

###Refining our hyperparameters

We can now refine our search of a good hyperparameter (the number of nodes in the hidden layer) by checking the value we've chosen from our previous comparison, with a value 2x smaller than it, and a value 2x larger than it.

![Comparison 2](/images/Julia_2_blog_2.png)

All three of our hyperparameters now cause the validation to level off at the same point, which means that the lowest (and so fastest) of our choices is the best one. We now repeate the test again, this time with the values 25, 50 and 100.

![Comparison 3](/images/Julia_2_blog_3.png)

This time, we can see that the validation accuracy for the network with 25 hidden nodes levels off before the others. Since time isn't a big factor (the 25 node network and the 50 node network took about the same amount of time to run), we'll use the 50 node hidden layer. We'll use this process of choosing hyperparameters again several times, as the network gets larger and more complex.

### The complete script

Our final script for this post can be found [here](/blog/Julia_2.py).

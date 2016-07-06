#L2 regularisation

###Overview

In this post, we're going to add [L2 regularisation](http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization) to our network. This involves tweaking our loss to keep weights small, and will help to lower overfitting, and help to raise our validation accuracy.

###Changing the loss

The main bulk of the change will happen in our ```getBatchLoss()``` function, as this is where the regularisation is applied.

```python
def getBatchLoss(tfLogits, tfBatchLabels, tfWeights):
	'''Calculates the loss from the current batch of images'''
	tfBatchLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tfLogits, tfBatchLabels))
	tfL2Loss = sum([tf.nn.l2_loss(tfWeights[index]) for index in tfWeights])
	return tfBatchLoss + (l2RegParam * tfL2Loss)
```

Since we're now using ```tfWeights``` in our function, the line calling it also needs to change.

```python
	tfBatchLoss = getBatchLoss(tfLogits, tfBatchLabels, tfWeights)
```

And finally we're going to need to find a value for our ```l2RegParam```.

###Finding the l2RegParam value

We've got our two sets of comparisons again.

![Comparison 1](/images/Julia_3_blog_1.png)

From which we select the 0.001 value to continue with.

![Comparison 2](/images/Julia_3_blog_2.png)

And from that, we pick 0.0005, as we want to use the smallest possible value possible, as it allows for the most accuracy. We can show this value in Python as ```5e-4```.

###The full script

The full script can be found [here](/blog/Julia_3.py). Note that I've changed some of the code so that the training process can be called from another script, to allow for easier comparisons of hyperparameters.

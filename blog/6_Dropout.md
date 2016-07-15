#Dropout

###Overview

We're going to see how to add dropout to our network, as well as it's effects on our network's accuracy.

###Adding dropout

To add dropout, we're first going to change the code to calculate logits. This is done by adding a ```tf.nn.dropout()``` to the outputs of our hidden layers.

```python
def getLogits(tfBatchDataset, tfWeights, tfBiases, dropout=1.0):
	'''Runs images through the network and returns logits'''
	tfHidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tfBatchDataset, tfWeights['hidden1']) + tfBiases['hidden1']), dropout)
	tfHidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(tfHidden1, tfWeights['hidden2']) + tfBiases['hidden2']), dropout)
	tfLogits = tf.matmul(tfHidden2, tfWeights['output']) + tfBiases['output']
	return tfLogits
```

Since we only want to add dropout when we're training, and not when we're testing our network. We've thus added the ```dropout=1.0``` argument. Note that when dropout is set to 1.0, there is no dropout, and when the dropout is set to 0.0, it means that all of the connection weights are set to zero, which is fairly counterintuitive. When we're training our network, we now specify the dropout.

```python
	tfLogits = getLogits(tfBatchDataset, tfWeights, tfBiases, dropout=dropout)
```

Finally we need to specify what our chosen dropout rate is. Since this is a new hyperparameter, we'll need to find the best value for it. Thankfully, we know that the value has to be between 0 and 1.

###Finding the dropout rate


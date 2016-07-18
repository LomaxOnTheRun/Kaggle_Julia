#Using momentum

###Overview

In the last post we got good results, but a very unstable accuracy. In this post, we'll look at how to use momentum to help keep our network accuracy more stable.

###Adding momentum

To add momentum to our system, we need to change our ```tf.train.GradientDescentOptimizer``` with a ```tf.train.MomentumOptimizer```.

```python
	# Optimizer
	tfOptimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(tfBatchLoss)
```

As you can see, we're still using our learning rate, and we're still minimising the batch loss, but we're now also utilising momentum, our newest hyperparameter.

To save us some time, we're going to follow the internet's advice, and set our momentum to 0.9, which gives us the fiollowing graph:

![Graph 1](/images/Julia_7_blog_1.png)


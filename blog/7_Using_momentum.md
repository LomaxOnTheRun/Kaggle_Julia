# Using momentum

### Overview

In the last post we got good results, but a very unstable accuracy. In this post, we'll look at how to use momentum to help keep our network accuracy more stable.

### Adding momentum

To add momentum to our system, we need to change our ```tf.train.GradientDescentOptimizer``` with a ```tf.train.MomentumOptimizer```.

```python
	# Optimizer
	tfOptimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(tfBatchLoss)
```

As you can see, we're still using our learning rate, and we're still minimising the batch loss, but we're now also utilising momentum, our newest hyperparameter.

To save us some time, we're going to follow the internet's advice, and set our momentum to 0.9, which gives us the fiollowing graph:

![Graph 1](/images/Julia_7_blog_1.png)

Well that looks awful. Let's have a think about why this might have happened. Momentum works by adding a fraction (in our case 0.9) of the previous step the network took, and simply adding it on the the current step we're taking. One of the common analogies with gradient descent is to imagine a ball rolling down down the side of a valley. Using momentum means that when the gradient is long and steep, we'll move much more quickly towards the bottom. However, using the same analogy, once the ball gets to the bottom of the valley, it will still have the momentum from the previous step, and so will overshoot the minima. One of the tings we can do then, is to reduce the step size, so that even when the ball is going at full speed, it'll be moving slower than before, and so won't overshoot by as much.

Since our results have turned into nonsence, we'll run our 10x comparisons again.

![Graph 2](/images/Julia_7_blog_2.png)

We can clearly see that while our value of 0.4 doesn't work, reducing it by a factor of 10, or even 100, does lead to it straightening back out again. It also looks like the 0.004 run didn't get a chance to top out its accuracy, so we're going to run a couple more tests, but this time we'll run them for 3000 steps. Since we're not sure which line is better out of the 0.04 and 0.004, we'll run a set of tests for each one and see what happens.

![Graph 3](/images/Julia_7_blog_3.png)

![Graph 4](/images/Julia_7_blog_4.png)

So it looks like 0.02 is our best value, and the one we'll procede with.

### The full script

You can get the full script for this post [here](/blog/Julia_7.py).

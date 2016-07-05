#Building our first neural network

###Overview

In this post, we'll be creating our first and most basic neural network, using TensorFlow. This is Google's contribution to the field of open-sourced neural network libraries. It uses a C++ backend, which allows for faster computations, although it can make it trickier to use to begin with, and more difficult to implement and tweak your own training algorithms later on. However, for our purposes it will do very well.

I should mention now that there is a lot going on in this script. As with the last post, I'll go through and explain each part as I go, but there are quite a few aspect of this that we need to look at to create a coherent and useful script. There is a link at the bottom to the finished script, so feel free to play around with it first, then come back and read up on any parts that don't make sense. The script will be roughly broken down into 4 parts:

1. Loading the pickled data
2. Setting up the network
3. Running the network
4. Plotting the network accuracy

###Network structure

For our initial foray into neural networks, we'll stick with an extremely simple structure of just 2 layers:

- **The input layer.** This is simply how we will 'show' the network our images. In this case, it will be a layer 400 nodes wide, one for each pixels in the image.
- **The output layer.** This is the layer from which we get the networks's assesment of the picture. There are as many output nodes as label IDs, and the output of each node shows how strongly the network believes the image we showed it has a particular ID. So the higher the output of node 10 is, the more strongly the network believes the image is of an '**A**'. Every output node is connected to every input node, and every one of these connections has a weight associated to it. Each node in the output layer also has a bias applied to it, and it is by adjusting these weights and biases that the network 'learns'.

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

Next we're going to define some constants, namely each of the datasets. The difference between placeholders and constants is that once the session starts, we cannot change a constant.

```python
	# Full datasets
	tfTrainDataset = tf.constant(trainDataset)
	tfValidDataset = tf.constant(validDataset)
	tfTestDataset = tf.constant(testDataset)
```

We're now going to build matrices for the weights and biases of the output layer. Again, because we're using TensorFlow, we can't just create a Numpy array. We're instead going to use two TensorFlow's own methods for creating matrices, one for creating a matrix of small randomised variables, and one for creating a matrix populated by zeros. We're also going to assign them as variables. The difference between a variable and a placeholder is that, while it can change value (unlike a constant), we cannot set a new value for it ourselves. We do this by calling the function ```getWeightsAndBiases()```...

```python
	# Network weights and biases
	tfWeights, tfBiases = getWeightsAndBiases()
```

... which looks like this:

```python
def getWeightsAndBiases():
	'''Create initial weights and biases'''
	tfWeights = tf.Variable(tf.truncated_normal([imageSize * imageSize, numLabels]))
	tfBiases = tf.Variable(tf.zeros([numLabels]))
	return tfWeights, tfBiases
```

We next get the logits output by our network. A logit is the 'raw' output of a node. You get it my multiplying the inputs into the node with the weights the node has for each of those inputs. Here, the inputs into the node refer to the pixels of the image. We can also carry out this process for a number of images simulatneously by mutiplying a matrix of 'stacked' images by a weight matrix of all the output nodes. We do exactly this by using the ```tf.matmul()``` expression. We then add a vector of biases to this outcome to get our 'raw output' or *logits*. This is called by:

```python
	# Training computation
	tfLogits = getLogits(tfBatchDataset, tfWeights, tfBiases)
```

which is defined as:

```python
def getLogits(tfBatchDataset, tfWeights, tfBiases):
	'''Runs images through the network and returns logits'''
	tfLogits = tf.matmul(tfBatchDataset, tfWeights) + tfBiases
	return tfLogits
```

The next bit requires a little more explation, as there are a few ideas to consider:

1. **Softmax.** Since we want an estimation of much the network believes a particular ID is the correct one, we want to have all output values range between 0 and 1. We also obviously want higher values to saty higher and lower values to stay lower. One way of doing this is to use the [softmax function] (https://en.wikipedia.org/wiki/Softmax_function).
2. **Cross-entropy.** This is basically a way of measuring how accurate our outputs are. It looks at the difference between each output value, and each value in the label vector we created above. So for every incorrect ID, the label vector value is 0, and the cross-entropy is bigger the higher our networks's evaluation of that ID is.
3. **Loss.** This is the value calculated by the cross-entropy, showing how inaccurate our network is for the images provided.

The ```tf.reduce_mean()``` function is TensorFlow's way of getting the mean value of a matrix. In our function this is called by:

```python
	tfBatchLoss = getBatchLoss(tfLogits, tfBatchLabels)
```

which is defined by:

```python
def getBatchLoss(tfLogits, tfBatchLabels):
	'''Calculates the loss from the current batch of images'''
	tfBatchLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tfLogits, tfBatchLabels))
	return tfBatchLoss
```

The optimiser is our choice of gradient descent method. We will run this every for every step, and it will use the learning rate given to try to minimise the loss calculated above. It will also automatically update the weights and biases of our network for us, which is pretty neat. We're going to go with a ```GradientDescentOptimizer``` as it's conceptually the simplest to work with.

```python
	# Optimizer
	tfOptimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(tfBatchLoss)
```

Finally, we're going to set up a quick way of getting the predictions for datasets whenever we want them. This is going to be useful as we want to be able to keep an eye on how well our network is performing.

```python
	# Predictions for the training, validation, and test datasets
	tfBatchPrediction = tf.nn.softmax(tfLogits)
	tfTrainPrediction = tf.nn.softmax(getLogits(tfTrainDataset, tfWeights, tfBiases))
	tfValidPrediction = tf.nn.softmax(getLogits(tfValidDataset, tfWeights, tfBiases))
	tfTestPrediction = tf.nn.softmax(getLogits(tfTestDataset, tfWeights, tfBiases))
```

A quick note on variable names. You'll probably notice that I've prefaced all of the variables used by TensorFlow with a *tf*. This is to help us remember that we can't just treat them like regular Python variables.

###Running the network

The bulk of the code to run the network will be in the same function as the code to set up the network, as we're going to need to be able to reference the variables we defined.

We begin by starting the session, which essentially kicks off the C++ code running in the background. We then initialise all the variables we defined in our network (our weights and biases). We're also going to keep track of the time taken between progress checks, to get a feel for how long it's taking our system to learn. Finally, we print out the headers for the progress table we're going to update as we go.

```python
	# Start the TensorFlow session
	with tf.Session() as session:
		
		# Initialise all the network variables
		tf.initialize_all_variables().run()
		
		# Start the timer and show info headings
		startTime = time.time()
		print '\nStep\tBatch loss\tTrain acc\tValid acc\tTime\n'
```

We're now going into what we're doing for every step of the training. Firstly, we need to get a new batch of images and associated labels. I'll go into this more later. Next, we need to create our ```feedDictionary```, which we're going to give to our network. Remember the placeholders we created? This is where we assign what their next values are going to be.

```python
		for step in xrange(numSteps):
		
			# Randomly get batch data, then feed it to the network
			batchDataset, batchLabels = getBatchData()
			feedDictionary = {tfBatchDataset : batchDataset, tfBatchLabels : batchLabels}
```

Next we're actually going to run our network and update the weights and biases. We do this by calling ```session.run()```, which tell TensorFlow that we want it to evaluate one or more of the variables we defined. The values of those variables are then returned to us. We use this calling mechanism to run and update our network by asking TensorFlow to evaluate ```tfOptimizer```, as this causes TensorFlow to run the ```tf.train.GradientDescentOptimizer(learningRate).minimize(tfBatchLoss)``` line, thus updating our network. Since we don't actually care about the output, we leave it blank by assigning the output to ```_```. We also get the loss for the batch data, which we'll show in our progress table, as well as pass in the feedDictionary to the network.

```python
			# Run the optimiser
			_, batchLoss = session.run([tfOptimizer, tfBatchLoss], feed_dict=feedDictionary)
```

We then get to out progress checking code. This is where we calculate and store the accuracies for both the test and the validation data. Calculating the accuracies is done in the ```accuracy()``` function, which we'll look at in a bit.

```python
			# Show updates every once in a while
			if (step % progressCheckInterval == 0):
				
				# Calculate training and validation accuracies
				trainAccuracy = accuracy(tfTrainPrediction.eval(), trainLabels)
				validAccuracy = accuracy(tfValidPrediction.eval(), validLabels)
				
				# Store accuracies for plotting
				plots['trainAccuracy'].append(trainAccuracy)
				plots['validAccuracy'].append(validAccuracy)
```

We then add a line to our progress table, and reset the timer so that we keep track of how long each group of steps have taken, rather than the cumulative time taken from the start.

```python
				# Show progress info
				timeTaken = time.time() - startTime
				print '%d\t%.3f\t\t%.1f%%\t\t%.1f%%\t\t%.3fs' % (step, batchLoss, trainAccuracy, validAccuracy, timeTaken)
				startTime = time.time()
```

Finally, we show how well the final version of our network does at predicting the test data.

```python
		# Print final accuracy of test dataset
		print "\nTest accuracy: %.1f%%\n" % accuracy(tfTestPrediction.eval(), testLabels)
```

###Getting the batch data

There are a number of ways of doing this, but we're going to randomly select a ```batchSize``` amount of data from our training dataset as our batch data.

```python
def getBatchData():
	'''Get batch data using randomly selected indexes'''
	randomIndexes = np.random.randint(trainDataset.shape[0], size=batchSize)
	batchDataset = trainDataset[randomIndexes]
	batchLabels = trainLabels[randomIndexes]
	return batchDataset, batchLabels
```

###Calculating accuracies

We first check to see if the network has found the correct answer, by looking at which ID the network has assigned the highest probability to, then we find the percentage of correct answers the networks has worked out for the dataset in question.

```python
def accuracy(predictions, labels):
	'''Check if most likely network outcomes are correct'''
	maxPredictions = np.argmax(predictions, 1)
	maxLabels = np.argmax(labels, 1)
	numCorrectPredictions = np.sum(maxPredictions == maxLabels)
	return (100.0 * numCorrectPredictions / predictions.shape[0])
```

###Running the program

We're now going to run the program, but with a slight twist. We're going to make sure that even if we get bored with our current run, we can still get a graph of what we've done so far. We do this by listening for a ```KeyboardInterrupt```, and dealing with it nicely.

```python
# Network hyperparameters
batchSize = 100
learningRate = 0.1
numHidden = 50

# Training parameters
numSteps = 10001
progressCheckInterval = 500

# Train the data
try:
	plots = {'trainAccuracy' : [], 'validAccuracy' : []}
	trainNeuralNetwork(plots)

# Deal with KeyboardInterrupts
except KeyboardInterrupt:
	print 'KeyboardInterrupt'

# Always show a plot, even if we've interrupted the training
finally:
	plotGraph(plots)
```

###Plotting the accuracies

We can now plot how well our network did at learning to read our images. This is going to be particuarly useful for analysis, as comparing graphs is much easier and more inutitive than looking at long tables of numbers.

```python
def plotGraph(plots):
	xAxisTrain = range(0, progressCheckInterval * len(plots['trainAccuracy']), progressCheckInterval)
	xAxisValid = range(0, progressCheckInterval * len(plots['validAccuracy']), progressCheckInterval)
	trainPlot, = plt.plot(xAxisTrain, plots['trainAccuracy'], label='Training data')
	validPlot, = plt.plot(xAxisValid, plots['validAccuracy'], label='Validation data')
	plt.legend(handles=[trainPlot, validPlot], loc=2)
	plt.xlabel('Number of steps taken')
	plt.ylabel('Accuracy (%)')
	plt.title('Julia 1\n')
	plt.tight_layout()
	plt.show()
```

###Example run

This is an example run of our brand new network, running on my small and not so powerful laptop:

```
Step	Batch loss	Train acc	Valid acc	Time

0		18.715		1.4%		1.7%		0.230s
500		8.324		3.7%		2.9%		3.877s
1000	6.738		5.5%		4.8%		3.606s
1500	6.512		6.9%		5.2%		3.014s
2000	5.524		8.1%		5.7%		3.776s
2500	5.579		9.3%		6.4%		3.382s
3000	4.915		11.0%		6.3%		2.852s
3500	4.738		12.2%		6.3%		2.950s
4000	4.519		13.1%		6.1%		2.808s
4500	4.086		13.4%		6.2%		3.460s
5000	3.844		14.5%		6.3%		2.893s
5500	4.316		15.4%		6.2%		2.860s
6000	3.934		16.9%		6.1%		3.108s
6500	3.804		17.7%		5.9%		3.631s
7000	3.396		18.2%		6.1%		3.300s
7500	3.731		19.1%		5.9%		3.244s
8000	3.412		19.9%		6.5%		3.295s
8500	3.790		20.7%		6.5%		3.287s
9000	3.633		21.2%		6.4%		2.867s
9500	3.728		21.7%		6.8%		3.045s
10000	3.186		22.0%		6.4%		3.194s

Test accuracy: 8.6%
```

And this is the graph produced:

![Julia_1.png](/images/Julia_1_blog.png)

As you can see, the accuracy we manage to get for our test data keeps improving, while the accuracy for our validation data stops improving relatively early. This is because our simple system begins to learn our training dataset and so gets better at identifying images from it, while it isn't good enough to generalise this knowledge to images it has never seen before. This is called overfitting.

###The full script

The full script for what we've seen today can be found [here] (/blog/Julia_1.py).

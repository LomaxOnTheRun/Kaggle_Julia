import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Fixed variables
imageSize = 20
numLabels = 62
batchSize = numLabels
numChannels = 1

# Network hyperparameters
learningRate = 0.4
numHidden = 50
l2RegParam = 5e-4
patchSize = 5
depth = 16

# Training parameters
numSteps = 2001
progressCheckInterval = 50

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

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, imageSize, imageSize, numChannels)).astype(np.float32)
	labels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# Reformat the data
trainDataset, trainLabels = reformat(trainDataset, trainLabels)
validDataset, validLabels = reformat(validDataset, validLabels)
testDataset, testLabels = reformat(testDataset, testLabels)

def putImagesInBins(images, labels):
	imageBins = [[] for i in xrange(62)]
	for index, image in enumerate(images):
		label = np.argmax(labels[index])
		imageBins[label].append(image)
	return imageBins

imageBins = putImagesInBins(trainDataset, trainLabels)

def trainNeuralNetwork(plots):
	'''Run the TensorFlow network'''

	# Placeholders for batch data
	tfBatchDataset = tf.placeholder(tf.float32, shape=(batchSize, imageSize, imageSize, numChannels))
	tfBatchLabels = tf.placeholder(tf.float32, shape=(batchSize, numLabels))
	
	# Full datasets
	tfTrainDataset = tf.constant(trainDataset)
	tfValidDataset = tf.constant(validDataset)
	tfTestDataset = tf.constant(testDataset)
  
	# Network weights and biases
	tfWeights, tfBiases = getWeightsAndBiases()
  
	# Training computation
	tfLogits = getLogits(tfBatchDataset, tfWeights, tfBiases)
	tfBatchLoss = getBatchLoss(tfLogits, tfBatchLabels, tfWeights)
  
	# Optimizer
	tfOptimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(tfBatchLoss)

	# Predictions for the training, validation, and test datasets
	tfBatchPrediction = tf.nn.softmax(tfLogits)
	tfTrainPrediction = tf.nn.softmax(getLogits(tfTrainDataset, tfWeights, tfBiases))
	tfValidPrediction = tf.nn.softmax(getLogits(tfValidDataset, tfWeights, tfBiases))
	tfTestPrediction = tf.nn.softmax(getLogits(tfTestDataset, tfWeights, tfBiases))
	
	# Start the TensorFlow session
	with tf.Session() as session:
		
		# Initialise all the network variables
		tf.initialize_all_variables().run()
		
		# Start the timer and show info headings
		startTime = time.time()
		print '\nStep\tBatch loss\tTrain acc\tValid acc\tTime\n'
		
		for step in xrange(numSteps):
		
			# Randomly get batch data, then feed it to the network
			batchDataset, batchLabels = getBatchData(step)
			feedDictionary = {tfBatchDataset : batchDataset, tfBatchLabels : batchLabels}
			
			# Run the optimiser
			_, batchLoss = session.run([tfOptimizer, tfBatchLoss], feed_dict=feedDictionary)
			
			# Show updates every once in a while
			if (step % progressCheckInterval == 0):
				
				# Calculate training and validation accuracies
				trainAccuracy = accuracy(tfTrainPrediction.eval(), trainLabels)
				validAccuracy = accuracy(tfValidPrediction.eval(), validLabels)
				
				# Store accuracies for plotting
				plots['trainAccuracy'].append(trainAccuracy)
				plots['validAccuracy'].append(validAccuracy)
				
				# Show progress info
				timeTaken = time.time() - startTime
				print '%d\t%.3f\t\t%.1f%%\t\t%.1f%%\t\t%.3fs' % (step, batchLoss, trainAccuracy, validAccuracy, timeTaken)
				startTime = time.time()
		
		# Print final accuracy of test dataset
		print("\nTest accuracy: %.1f%%\n" % accuracy(tfTestPrediction.eval(), testLabels))

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

def getLogits(tfBatchDataset, tfWeights, tfBiases, dropout=1.0):
	'''Runs images through the network and returns logits'''
	tfConv1 = tf.nn.relu(tf.nn.conv2d(tfBatchDataset, tfWeights['conv1'], [1, 1, 1, 1], padding='SAME') + tfBiases['conv1'])
	tfShape = tfConv1.get_shape().as_list()
	tfReshape = tf.reshape(tfConv1, [tfShape[0], tfShape[1] * tfShape[2] * tfShape[3]])
	tfHidden1 = tf.nn.relu(tf.matmul(tfReshape, tfWeights['hidden1']) + tfBiases['hidden1'])
	tfHidden2 = tf.nn.relu(tf.matmul(tfHidden1, tfWeights['hidden2']) + tfBiases['hidden2'])
	tfLogits = tf.matmul(tfHidden2, tfWeights['output']) + tfBiases['output']
	return tfLogits

def getBatchLoss(tfLogits, tfBatchLabels, tfWeights):
	'''Calculates the loss from the current batch of images'''
	tfBatchLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tfLogits, tfBatchLabels))
	tfL2Loss = sum([tf.nn.l2_loss(tfWeights[index]) for index in tfWeights])
	return tfBatchLoss + (l2RegParam * tfL2Loss)

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

def accuracy(predictions, labels, showGuesses=False):
	'''Check if most likely network outcomes are correct'''
	if showGuesses:
		guesses = []
		for index, prediction in enumerate(predictions):
			guess = np.argmax(prediction)
			if (guess == np.argmax(labels[index])):
				guesses.append(guess)
		print '\n%s\n' % guesses
		return (100.0 * len(guesses)) / predictions.shape[0]
	else:
		maxPredictions = np.argmax(predictions, 1)
		maxLabels = np.argmax(labels, 1)
		numCorrectPredictions = np.sum(maxPredictions == maxLabels)
		return (100.0 * numCorrectPredictions / predictions.shape[0])

def plotGraph(plots):
	xAxisTrain = range(0, progressCheckInterval * len(plots['trainAccuracy']), progressCheckInterval)
	xAxisValid = range(0, progressCheckInterval * len(plots['validAccuracy']), progressCheckInterval)
	trainPlot, = plt.plot(xAxisTrain, plots['trainAccuracy'], label='Training data')
	validPlot, = plt.plot(xAxisValid, plots['validAccuracy'], label='Validation data')
	plt.legend(handles=[trainPlot, validPlot], loc=2)
	plt.xlabel('Number of steps taken')
	plt.ylabel('Accuracy (%)')
	plt.title('Julia 6\n')
	plt.tight_layout()
	plt.show()

def run(showGraph=True):
	# Train the data
	try:
		plots = {'trainAccuracy' : [], 'validAccuracy' : []}
		trainNeuralNetwork(plots)

	# Deal with KeyboardInterrupts
	except KeyboardInterrupt:
		print 'KeyboardInterrupt'

	# Always show a plot, even if we've interrupted the training
	finally:
		if showGraph:
			plotGraph(plots)
	
	return plots

# Run the script
if __name__ == '__main__':
	run()

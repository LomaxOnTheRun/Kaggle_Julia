import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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

def checkShapes():
	print 'Training set:', trainDataset.shape, trainLabels.shape
	print 'Validation set:', validDataset.shape, validLabels.shape
	print 'Test set:', testDataset.shape, testLabels.shape

# Uncomment these to run sanity checks
##checkShapes()

# Fixed variables
imageSize = 20
numLabels = 62

# Hyperparameters
numSteps = 10001
progressCheckInterval = 500
batchSize = 100
learningRate = 0.1

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, dataset.shape[1] * dataset.shape[2])).astype(np.float32)
	labels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# Reformat the data
trainDataset, trainLabels = reformat(trainDataset, trainLabels)
validDataset, validLabels = reformat(validDataset, validLabels)
testDataset, testLabels = reformat(testDataset, testLabels)

##checkShapes()

def trainNeuralNetwork(plots):
	'''Run the TensorFlow network'''

	# Placeholders for batch data
	tfBatchDataset = tf.placeholder(tf.float32, shape=(batchSize, imageSize * imageSize))
	tfBatchLabels = tf.placeholder(tf.float32, shape=(batchSize, numLabels))
	
	# Full datasets
	tfTrainDataset = tf.constant(trainDataset)
	tfValidDataset = tf.constant(validDataset)
	tfTestDataset = tf.constant(testDataset)
  
	# Network weights and biases
	tfWeights = tf.Variable(tf.truncated_normal([imageSize * imageSize, numLabels]))
	tfBiases = tf.Variable(tf.zeros([numLabels]))
  
	# Training computation
	tfLogits = tf.matmul(tfBatchDataset, tfWeights) + tfBiases
	tfBatchLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tfLogits, tfBatchLabels))
  
	# Optimizer
	tfOptimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(tfBatchLoss)

	# Predictions for the training, validation, and test datasets
	tfBatchPrediction = tf.nn.softmax(tfLogits)
	tfTrainPrediction = tf.nn.softmax(tf.matmul(tfTrainDataset, tfWeights) + tfBiases)
	tfValidPrediction = tf.nn.softmax(tf.matmul(tfValidDataset, tfWeights) + tfBiases)
	tfTestPrediction = tf.nn.softmax(tf.matmul(tfTestDataset, tfWeights) + tfBiases)
	
	# Start the TensorFlow session
	with tf.Session() as session:
		
		# Initialise all the network variables
		tf.initialize_all_variables().run()
		
		# Start the timer and show info headings
		startTime = time.time()
		print '\nStep\tBatch loss\tTrain acc\tValid acc\tTime\n'
		
		for step in xrange(numSteps):
		
			# Randomly get batch data, then feed it to the network
			batchDataset, batchLabels = getBatchData()
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

def getBatchData():
	'''Get batch data using randomly selected indexes'''
	randomIndexes = np.random.randint(trainDataset.shape[0], size=batchSize)
	batchDataset = trainDataset[randomIndexes]
	batchLabels = trainLabels[randomIndexes]
	return batchDataset, batchLabels

def accuracy(predictions, labels):
	'''Check if most likely network outcomes are correct'''
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
	plt.title('Julia 1\n')
	plt.tight_layout()
	plt.show()


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

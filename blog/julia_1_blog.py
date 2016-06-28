## Initial setup, no tweaking
##
## Minibatch loss at step 3000: 3.380355
## Minibatch accuracy: 17.0%
## Validation accuracy: 4.2%
## Test accuracy: 4.2%
## 
## I will be keeping track of my progress on GitHub
## -- This hasn't happened yet as I don't want to mess around with GitHub while on Rene's internet

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with open('julia.pickle', 'rb') as f:
	save = pickle.load(f)
	trainDataset = save['train_dataset']
	trainLabels = save['train_labels']
	validDataset = save['valid_dataset']
	validLabels = save['valid_labels']
	testDataset = save['test_dataset']
	testLabels = save['test_labels']
	del save  # Hint to help garbage collection free up memory

##	print 'Training set', train_dataset.shape, train_labels.shape
##	print 'Validation set', valid_dataset.shape, valid_labels.shape
##	print 'Test set', test_dataset.shape, test_labels.shape

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, dataset.shape[1] * dataset.shape[2])).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels



num_steps = 10001
batch_size = 100
image_size = 20
num_labels = 62

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

##print('Training set', train_dataset.shape, train_labels.shape)
##print('Validation set', valid_dataset.shape, valid_labels.shape)
##print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def runStochasticGradientDescent(plot_train_accuracy, plot_valid_accuracy):

	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	tf_train_dataset_full = tf.constant(train_dataset)
  
	# Variables.
	weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
  
	# Training computation.
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
	train_prediction_full = tf.nn.softmax(tf.matmul(tf_train_dataset_full, weights) + biases)
	
	# Run the optimiser.
	with tf.Session() as session:
		tf.initialize_all_variables().run()
		for step in range(num_steps):
			idx = np.random.randint(train_dataset.shape[0], size=batch_size) # Randomly select indexes for batch
			batch_data = train_dataset[idx]
			batch_labels = train_labels[idx]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
			if (step % 500 == 0):
				print("Minibatch loss at step %d: %f" % (step, l))
				print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
				train_accuracy = accuracy(train_prediction_full.eval(), train_labels)
				plot_train_accuracy.append(train_accuracy)
				print("Training total accuracy: %.1f%%" % train_accuracy)
				valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
				plot_valid_accuracy.append(valid_accuracy)
		print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


try:
	plot_train_accuracy = []
	plot_valid_accuracy = []
	runStochasticGradientDescent(plot_train_accuracy, plot_valid_accuracy)
except KeyboardInterrupt:
	print 'KeyboardInterrupt'
finally:
	plt.plot([500*i for i in range(len(plot_train_accuracy))], plot_train_accuracy)
	plt.plot([500*i for i in range(len(plot_valid_accuracy))], plot_valid_accuracy)
	plt.show()



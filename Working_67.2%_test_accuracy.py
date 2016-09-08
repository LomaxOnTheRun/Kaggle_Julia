## Initial setup
## - Added L2 regularization
## - Added hidden layer (10 x relu)
## - Added dropout (0.5)
## - Added a convolution layer
## - Added a pooling layer (quartered weights, same outcome [if not higher])
## - Added second convolution and pooling layer
## - Added a third convolution layer (which goes to 120 1x1 feature maps)
## - Set a changing learning rate
## - Augment images as I'm using them (rotation only)
## - Using momentum
## - Using a different optimiser (calculates momentum automatically)
## - Changing network layout
## - Added a second hidden layer

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from random import randint, choice
import time

pickle_file = 'julia.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# Network shape
image_size = train_dataset[0].shape[0]
num_channels = 1 # grayscale
depth1 = 16
depth2 = 32
depth3 = 64
patch1_size = 5
patch2_size = 3
patch3_size = 3
num_hidden1 = 64
num_hidden2 = 64
num_labels = 62

# Network parameters
batch_size = 62
num_steps = 10001
l2_reg_param = 5e-4
dropout = 0.5

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


def tweakImages(batchImages):
	for index, image in enumerate(batchImages):
		# Change shape for tweaking
		image = np.reshape(image, (image.shape[0], image.shape[1]))
		original2DShape = image.shape
		# Rotate image
		image = ndimage.rotate(image, randint(-10, 10), reshape=False)
		# Shift image
		image = ndimage.shift(image, (randint(-1, 1), randint(-1, 1)))
		# Invert image
		if choice([True, False]):	image = -image
		# Spline (smoothing of neighbouring pixels)
		if choice([True, False]):	image = ndimage.spline_filter(image)
		# Put back into shape
		image = np.reshape(image, (image.shape[0], image.shape[1], 1))
		batchImages[index] = image
	return batchImages

def show(x):
	x = np.reshape(x, (x.shape[0], x.shape[1]))
	plt.imshow(x)
	plt.show()

def accuracy(predictions, labels, showGuesses=False):
	if showGuesses:
		guesses = []
		for index, p in enumerate(predictions):
			guess = np.argmax(p)
			if (guess == np.argmax(labels[index])):
				guesses.append(guess)
		print guesses
		return (100.0 * len(guesses)) / predictions.shape[0]
	else:
		return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def runBatchGradientDescent(plot_train_accuracy, plot_valid_accuracy):

	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	tf_train_dataset_full = tf.constant(train_dataset)
  
	# Variables.
	w_conv1 = tf.Variable(tf.truncated_normal([patch1_size, patch1_size, num_channels, depth1], stddev=0.1))
	w_conv2 = tf.Variable(tf.truncated_normal([patch2_size, patch2_size, depth1, depth2], stddev=0.1))
	w_conv3 = tf.Variable(tf.truncated_normal([patch3_size, patch3_size, depth2, depth3], stddev=0.1))
	w_hidden1 = tf.Variable(tf.truncated_normal([1 * 1 * depth3, num_hidden1], stddev=0.1))
	w_hidden2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
	w_output = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
	weights = {'conv1' : w_conv1, 'conv2' : w_conv2, 'conv3' : w_conv3, 'hidden1' : w_hidden1, 'hidden2' : w_hidden2, 'output' : w_output}
	b_conv1 = tf.Variable(tf.zeros([depth1]))
	b_conv2 = tf.Variable(tf.zeros([depth2]))
	b_conv3 = tf.Variable(tf.zeros([depth3]))
	b_hidden1 = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))
	b_hidden2 = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
	b_output = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	biases = {'conv1' : b_conv1, 'conv2' : b_conv2, 'conv3' : b_conv3, 'hidden1' : b_hidden1, 'hidden2' : b_hidden2, 'output' : b_output}
	
	num_weights = sum([tf.size(weights[index]) for index in weights])
	
	# Model.
	def model(data, dropout=1.0):
		conv1 = tf.nn.conv2d(data, weights['conv1'], [1, 1, 1, 1], padding='VALID')
		conv1 = tf.nn.relu(conv1 + biases['conv1'])
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.conv2d(pool1, weights['conv2'], [1, 1, 1, 1], padding='VALID')
		conv2 = tf.nn.relu(conv2 + biases['conv2'])
		pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv3 = tf.nn.conv2d(pool2, weights['conv3'], [1, 1, 1, 1], padding='VALID')
		conv3 = tf.nn.relu(conv3 + biases['conv3'])
		shape = conv3.get_shape().as_list()
		reshape = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden1 = tf.nn.relu(tf.matmul(reshape, weights['hidden1']) + biases['hidden1'])
		hidden1_dropped = tf.nn.dropout(hidden1, dropout)
		hidden2 = tf.nn.relu(tf.matmul(reshape, weights['hidden2']) + biases['hidden2'])
		hidden2_dropped = tf.nn.dropout(hidden2, dropout)
		output = tf.matmul(hidden2_dropped, weights['output']) + biases['output']
		return output
  
	# Training computation.
	logits = model(tf_train_dataset, dropout=dropout)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	l2_loss = sum([tf.nn.l2_loss(weights[index]) for index in weights])
	total_loss = loss + (l2_reg_param * l2_loss)
    
	# Optimizer.
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.5, global_step, decay_steps=500, decay_rate=0.5, staircase=True)
	optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.9).minimize(total_loss, global_step=global_step)
##	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))
	train_prediction_full = tf.nn.softmax(model(tf_train_dataset_full))

	with tf.Session() as session:
		tf.initialize_all_variables().run()
		print '\nNum weights:', session.run(num_weights)
		print '\nStep\tBatch loss\tValid acc\tTime\n'
		start_time = time.time()
		for step in range(num_steps):
			idx = np.random.randint(train_dataset.shape[0], size=batch_size)
			batch_data = tweakImages(train_dataset[idx])
			batch_labels = train_labels[idx]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l = session.run([optimizer, total_loss], feed_dict=feed_dict)
			if (step % 50 == 0):
##				print "Minibatch loss at step %d: %f" % (step, l)
##				train_accuracy = accuracy(train_prediction_full.eval(), train_labels)
##				plot_train_accuracy.append(train_accuracy)
##				print("Training total accuracy: %.1f%%" % train_accuracy)
				valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)#, showGuesses=True)
				plot_valid_accuracy.append(valid_accuracy)
##				print "Validation accuracy: %.1f%%" % valid_accuracy
##				print 'Learning rate = %.3f\tMomentum = %.3f' % (learn, mom)
				time_taken = time.time() - start_time
				print '%d\t%.5f\t\t%.1f%%\t\t%.3fs' % (step, l, valid_accuracy, time_taken)
				start_time = time.time()
		print('\nTest accuracy: %.1f%%\n' % accuracy(test_prediction.eval(), test_labels))

def showBatchGradientDescent():
	try:
		plot_train_accuracy = []
		plot_valid_accuracy = []
		runBatchGradientDescent(plot_train_accuracy, plot_valid_accuracy)
	except KeyboardInterrupt:
		print 'KeyboardInterrupt'
	finally:
		train_line, = plt.plot([500*i for i in range(len(plot_train_accuracy))], plot_train_accuracy, label='Training data')
		valid_line, = plt.plot([500*i for i in range(len(plot_valid_accuracy))], plot_valid_accuracy, label='Validation data')
		plt.legend(handles=[train_line, valid_line], loc=2)
		plt.xlabel('Number of epochs')
		plt.ylabel('Accuracy (%)')
		plt.title('BatchSize=%s -- LearningRate=%s -- Momentum=%s\nL2RegParam=%s -- Dropout=%s\nPatchSize1=%s -- PatchSize2=%s -- PatchSize3=%s\nDepth1=%s -- Depth2=%s -- Depth3=%s\nNumHidden1=%s -- NumHidden2=%s' % (batch_size, '0.5 / 2 every 500 steps', 'n/a', l2_reg_param, dropout, patch1_size, patch2_size, patch3_size, depth1, depth2, depth3, num_hidden1, num_hidden2))
		plt.tight_layout()
		plt.show()

showBatchGradientDescent()




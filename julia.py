## Initial setup, no tweaking
##
## Minibatch loss at step 3000: 3.380355
## Minibatch accuracy: 17.0%
## Validation accuracy: 4.2%
## Test accuracy: 4.2%

import pickle
import numpy as np
import tensorflow as tf

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

##	print 'Training set', train_dataset.shape, train_labels.shape
##	print 'Validation set', valid_dataset.shape, valid_labels.shape
##	print 'Test set', test_dataset.shape, test_labels.shape

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, dataset.shape[1] * dataset.shape[2])).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

num_steps = 3001
batch_size = 100
image_size = 20
num_labels = 62

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

##print('Training set', train_dataset.shape, train_labels.shape)
##print('Validation set', valid_dataset.shape, valid_labels.shape)
##print('Test set', test_dataset.shape, test_labels.shape)

def runStochasticGradientDescent():

	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
  
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
				print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
		print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

runStochasticGradientDescent()





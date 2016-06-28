import numpy as np
import csv
from scipy import ndimage
import os
import random
import pickle
import matplotlib.pyplot as plt


def getLabelId(label):
	
	# Calculate the ID for each character (0-9, then A-Z, then a-z), with 62 in total
	label = ord(label)
	if 48 <= label <= 57:		labelId = label - 48
	elif 65 <= label <= 90:		labelId = label - 55
	elif 97 <= label <= 122:	labelId = label - 61
	
	return labelId


def getLabels():

	# Get all labels from CSV file
	labels = []
	with open('trainLabels.csv') as f:
		reader = csv.DictReader(f)
		for line in reader:
			labels.append(line['Class'])
	
	# Change labels into integer bins
	labels = [getLabel(x) for x in labels]
	
	return labels


def getImageData():
	'''Creates a numpy array of the image dataset'''
	
	# Image info
	imageSize = 20
	pixelDepth = 255.0
	
	# Find how many images exist
	folder = 'misc/trainResized20x20'
	numImages = len(os.listdir(folder))
	
	# Create a single numpy array placeholder for the whole dataset
	datasetShape = (numImages, imageSize, imageSize)
	dataset = np.ndarray(shape=datasetShape, dtype=np.float32)
	
	for imageNum in xrange(1, numImages+1):
		
		# Get each image
		imageName = '%s.Bmp' % imageNum
		imageFile = os.path.join(folder, imageName)
		imageData = ndimage.imread(imageFile).astype(float)
		
		# Pre-process each image
		imageData = imageData - (pixelDepth / 2)	# Centre the colour values around zero
		imageData = imageData / pixelDepth			# Reduce the standard deviation to 1
		if len(imageData.shape) == 3:
			imageData = imageData.mean(axis=2)		# Flatten image into a 2D greyscale image
		
		# Put image into the numpy array
		dataset[imageNum, :, :] = imageData
    
	# Show dataset stats
	print 'Full dataset tensor:', dataset.shape
	print 'Mean:', np.mean(dataset)
	print 'Standard deviation:', np.std(dataset)
	
	return dataset


def shuffleDataAndLabels(data, labels):
	
	# Zip images and labels before shuffling so they don't get mixed up
	zipped = zip(data, labels)
	random.shuffle(zipped)
	data, labels = zip(*zipped)
	
	return data, labels


def splitData(data, labels, validationSize, testSize):
	
	# Split datasets
	testData       = np.asarray(data[:testSize], dtype=np.float32)
	validationData = np.asarray(data[testSize:testSize+validationSize], dtype=np.float32)
	trainingData   = np.asarray(data[testSize+validationSize:], dtype=np.float32)
	data = [trainingData, validationData, testData]
	
	# Split labels
	testLabels       = np.asarray(labels[:testSize], dtype=np.float32)
	validationLabels = np.asarray(labels[testSize:testSize+validationSize], dtype=np.float32)
	trainingLabels   = np.asarray(labels[testSize+validationSize:], dtype=np.float32)
	labels = [trainingLabels, validationLabels, testLabels]
	
	return data, labels


def saveData(pickleFile, (trainingData, validationData, testData), (trainingLabels, validationLabels, testLabels)):
	
	# Save each dataset / label set separately
	with open(pickleFile, 'wb') as f:
		save = {
			'train_dataset': trainingData,
			'train_labels': trainingLabels,
			'valid_dataset': validationData,
			'valid_labels': validationLabels,
			'test_dataset': testData,
			'test_labels': testLabels,
			}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	
	# Show final pickle file size
	print 'Compressed pickle size: %s' % os.stat(pickleFile).st_size


def pickleFiles():

	# Get all images and labels
	imageData = getImageData()
	labels = getLabels()

	# Shuffle them to randomise the training data
	imageData, labels = shuffleDataAndLabels(imageData, labels)

	# Select the validation and test data sizes
	# The rest will go to training (total number of images = 6283)
	validationSize = 1000
	testSize = 1000
	imageData, labels = splitData(imageData, labels, testSize, validationSize)
	
	# Save the numpy arrays to file
	saveData('julia_blog.pickle', imageData, labels)


pickleFiles()


def showImages(numImages):

	# Get images and labels to check against
	imageData = getImageData()
	labels = getLabels()
	
	# Show each image and label to check they match up correctly
	for i in xrange(numImages):
		print 'Label: %s' % labels[i]
		plt.imshow(imageData[i])
		plt.show()


##showImages()




#Getting the training data

###What?

In this blog entry we will be downloading, pre-processing and storing the training images for our network to use.

###How?

Kaggle handily offers a 'Data' page for each of its competitions. For this particular competition, there are two sets of training data: one set of raw pictures, each with varying heights and widths, and one that they've resized to 20x20 pixels. As I'm keeping things small, the resized images will do nicely for now. We can always play around with larger images once we've got a network we know works.

So, having gone [here] (https://www.kaggle.com/c/street-view-getting-started-with-julia/data), downloaded the 'trainResized.zip' file and extracted it, I now have a folder full of 20x20 images to work with. Add to that the 'trainLabels.csv' file and we are ready to go.

The first thing I'll need to do is to get the images in a format that is relatively quick for my network to load every time. There are a couple of ways of doing this, but I'm going to choose to 'pickle' them. This essentially puts the data for all the images in a compressed file, keeping their original Python and Numpy structures. We can break down the task into a few different parts. The first is to define a function which puts all of the images into a Numpy array, which the network will need in order to train itself.

```python
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
```

We then need to put the labels into a list so that we can match them to the images, and the network knows what a particular image is supposed to be. We do this by assigning each number and letter an ID, which will be integers ranging from 0 to 61. How you choose to assign the IDs is arbitrary, so long as each number and letter (upper and lower case) gets a unique ID between 0 and 61. I have chosen to assign numbers the IDs 0-9 (corresponding to their actual number), then the IDs 10-35 for upper case letters A-Z, and finally the IDs 36-61 for the lower case letters a-z.

```python
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
```

All that remains to be done is to shuffle the dataset, select the size of the validation and test datasets, and pickle the final datasets and labels. We can then access the datasets, and corresponding labels, any time we want to, without having to load in and pre-process all of the images each time.

```python
import numpy as np
import csv
from scipy import ndimage
import os
import random
import pickle
import matplotlib.pyplot as plt

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
```

One final thought: check your work. In this instance, it is particularly easy to get one of the datasets you've just created, and check that its associated label is the correct one. My first major headache with this project came with my failure to check exactly this, and I was left scratching my head for a couple of days, unable to work out why my network subbornly chose to always just guess the most abundant training letter (an 'A' in my case), regardless of what image it was shown. It had simply worked out that since there was no correlation between the image and the label, it would get the highest score by guessing the most abundant training label, the clever little thing. Here is the code to check the labels match the images.

```python
def showImages(numImages):

	# Get images and labels to check against
	imageData = getImageData()
	labels = getLabels()
	
	# Show each image and label to check they match up correctly
	for i in xrange(numImages):
		print 'Label: %s' % labels[i]
		plt.imshow(imageData[i])
		plt.show()


showImages()
```

###Why?

This is why.

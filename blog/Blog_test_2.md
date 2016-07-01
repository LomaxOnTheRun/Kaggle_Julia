#Getting the training data

###Overview

In this session we will download the training data from Kaggle, pre-process it, and then save it to a Pickle file using Python.

###Downloading the data

Kaggle handily offers a 'Data' page for each of its competitions, where you can get the training data, test data, and a few other useful bits and pieces. For this particular competition, there are two sets of training data: one set of raw pictures, each with varying heights and widths, and one that they've resized to 20x20 pixels. As I'm keeping things small to start with, the resized images will do nicely for now. We can always play around with larger images once we've got a network we know works.

So, having gone [here] (https://www.kaggle.com/c/street-view-getting-started-with-julia/data), downloaded the 'trainResized.zip' file and extracted it, we now have a folder full of 20x20 images to work with. Add to that the 'trainLabels.csv' file and we have everything we need for now. In terms of file structure, I keep my image folders and scripts inside the same folder. You don't have to do the same, but you may need to chage the folder names in the code if you decide to use a different folder structure.

###A look at the script

We're now ready to crack out our trusty Python and get on with some actual coding. The first thing we'll do is to create a very high level function, which we'll fill in as we go.

```python
def pickleFiles():
	'''Gets, splits and saves images and corresponding labels'''

	# Get all images and labels
	imageData = getImageData()
	labels = getLabels()

	# Set validation and test dataset sizes, the rest will be for training (total images = 6283)
	validSize = 1000
	testSize = 1000
	imageData, labels = splitData(imageData, labels, testSize, validSize)
	
	# Save the numpy arrays to file
	saveData('julia.pickle', imageData, labels)
```

This is obviously not yet code we can run - we still need to define all the functions we call. Nonetheless, it's useful insofar as it gives us a clear idea of what we need to do. Just before we get down to creating all of the required functions, we'll take a quick look at why we need validation and test datasets, as well as our training dataset.

###Why do we need validation *and* test datasets?

The reason for having a test dataset is very simple: we need to know how good our network is at identifying images it has *never seen before*. Simply checking how well our network is performing against its training data isn't an accurate test of this, so we need to keep back some images for testing.

The reason for having a validation dataset is slightly more subtle. Whilst we won't ever show the network any of the images from the validation dataset either, we will be using it to monitor the progress of our network, and to change the hyper-parameters of the network accordingly. In this way, some of the information about the validation dataset will bleed through to the network. To stop the same thing from happening to our test dataset, we will only ever run our network against it once, after it has completely finished training.

###Getting the image data

The first of the functions we need to define is ```getImageData()```, which will go through all of the 20x20 images in our trainResized folder, and turn them into a Numpy array. The image files are named 'X.Bmp' where X is a number going from 1 to 6283, which makes it very easy for us to load them in order. Keeping them in this order is very important, as we need them to match up with their corresponding labels in the CSV file.

Most of the images we're reading in are represented by 3D matrices, which can be thought of as a stack of 3 versions of the same image; one red, one green, one blue. Since our network doesn't need to know anything about the colour of the image, we're going to 'flatten' the 3D matrix into a 2D matrix, effectively turning the colour image into a greyscale one. This has two main advantages:

1. The network has less information per image to worry about, which will make it faster at processing each image.
2. All of the 2D images can be 'stacked' on top of one another to create one large 3D matrix, which represents one complete dataset. This 3D matrix is what our function is going to return.

```python
import os
import numpy as np
from scipy import ndimage

def getImageData():
	'''Creates a numpy array of the image dataset'''
	
	# Image info
	imageSize = 20
	pixelDepth = 255.0
	
	# Find how many images exist
	folder = 'trainResized'
	numImages = len(os.listdir(folder))
	
	# Create a single numpy array placeholder for the whole dataset
	datasetShape = (numImages, imageSize, imageSize)
	dataset = np.ndarray(shape=datasetShape, dtype=np.float32)
	
	for imageNum in xrange(numImages):
		
		# Get each image
		imageName = '%s.Bmp' % (imageNum+1)
		imageFile = os.path.join(folder, imageName)
		imageData = ndimage.imread(imageFile).astype(float)
		
		# Pre-process each image
		if len(imageData.shape) == 3:
			imageData = imageData.mean(axis=2)		# Flatten image into a 2D greyscale image
		imageData = imageData - (pixelDepth / 2)	# Centre the colour values around zero
		imageData = imageData / (pixelDepth / 2)	# Reduce the standard deviation to 1
		
		# Put image into the numpy array
		dataset[imageNum, :, :] = imageData
    
	# Show dataset stats
	print 'Full dataset tensor:', dataset.shape
	print 'Mean:', np.mean(dataset)
	print 'Standard deviation:', np.std(dataset)
	
	return dataset
```

Unlike our high level script, this one can actually be run as it is. You just need to add ```getImageData()``` to the bottom of your script, and you should see the dataset stats printed out when you run it.

###What's with the pre-processing stuff?

In addition to flattening the images, we've also carried out two further actions:

1. *We centred the values of the pixels around zero.* We've done this to better 'define' the problem. Gradient descent methods (which we'll be using) have a much easier time working with well-defined problems, which will manifest itself as a faster learning rate and more accurate results. **(MORE INFO AND PICS?)**

2. *We restricted the range of values the pixels can take to between -1 and 1.* This will help us to reduce floating point errors. When dealing with floating point numbers (e.g. Python's ```float``` numbers), small rounding errors occur as the processor is not able to hold infinitely long decimal places in memory. This only really starts becoming a problem when trying to calculate tiny differences between very large numbers and, while 255 may not look very large, it becomes problematic when we use large networks with very large matrices. Remember, we'll be trying to squeeze out every possible bit of accuracy from our network, so even a fraction of a percent will make a difference.

###Getting the labels

We can now pull our labels out from the 'trainLabels.csv' we've already downloaded.

```python
import csv

def getLabels():
	'''Returns the alphanumeric labels from trainLabels.csv'''

	# Get all labels from CSV file
	labels = []
	with open('trainLabels.csv') as f:
		reader = csv.DictReader(f)
		for line in reader:
			labels.append(line['Class'])
	
	return labels
```

If you run this script and take a look at the list this returns, you'll see it's full of one character strings with labels 0-9, A-Z and a-z, all jumbled up. If you open the csv file using another program, you can also check by eye that the first few rows match. Now, since our network can't handle alphanumeric labels, we need to turn all of them into purely numerical ones, which will each have a corresponding output node. We do this by assigning each alphanumeric label an ID, ranging between 0 and 61. How you decide to do this is somewhat arbitrary, as long as each label is given a unique ID in that range. I've chosen to allocate IDs sequentially to each number, then to each uppercase letter, then to each lowercase letter. The code now looks like this:

```python
import csv

def getLabels():
	'''Returns the numeric IDs of the labels from trainLabels.csv'''
	
	# Get all labels from CSV file
	labels = []
	with open('trainLabels.csv') as f:
		reader = csv.DictReader(f)
		for line in reader:
			labels.append(line['Class'])
	
	# Change labels into IDs
	labels = [getLabelId(label) for label in labels]
	
	return labels


def getLabelId(label):
	'''Returns the numeric ID for the given alphanumeric label'''
	
	# Calculate the ID for each character (0-9, then A-Z, then a-z), with 62 in total
	label = ord(label)
	if   ord('0') <= label <= ord('9'):		labelId = label - ord('0')
	elif ord('A') <= label <= ord('Z'):		labelId = label - ord('A') + 10
	elif ord('a') <= label <= ord('z'):		labelId = label - ord('a') + 36
	
	return labelId
```

Once again, you can run ```getLabels()``` in both its above forms, and get a list of either alphanumeric labels or numeric label IDs. Note that even once I turn the alphanumeric labels into their numeric IDs, I still choose to call them 'labels' in the code. This is because the alphanumeric labels and their numeric IDs are conceptually equivalent, and so will be used interchangably from here on out.

###Splitting the data

As we mention above, we need to split the data between training, validation and test datasets. I've chosen to do this by specifying the size of the validation and test datasets, and letting the training dataset be compaomised of all of the remaining images. Once again, how you split the data is up to you, but you must make sure that no dataset has any images from any of the other datasets.

```python
def splitData(data, labels, validationSize, testSize):
	'''Splits the data and labels into three separate datasets'''
	
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
```

One thing to note at this point is that I've assumed that the dataset we downloaded is randomly shuffled. You can elect to shuffle the data further still to make sure this is the case, but I've decided not to do this here as I wanted to have recreatable validation and test datasets. The reason for this is to enable you to get very similar accuracies to what I've gotten, by running the code I've posted.

###How much training / validation / test data do we need?

In an ideal world, as much of all three as possible. However, since that is almost never the case, we normally have to look at the best ways of divvying up the available data into training, validation and test datasets. The basic idea is that we want as much data as possible in our training dataset (allowing it to become as good as possible), whilst still having enough data in each of our other datasets to be able to get a good idea of the network's accuracy. The general rule of thumb is to have somewhere between 10% and 20% of your data in each of your validation and test sets.

###Saving the data

We now get to the final part of our script; saving our data as a Pickle file, also referred to as *pickling* the data. Pickling data allows us to store data in a compressed format, whilst keeping their Python and Numpy data structures.

```python
import pickle
import os

def saveData(pickleFile, imageData, labels):
	'''Pickles the image datasets and labels'''
	
	# Unpack data and labels
	trainingData, validationData, testData = imageData
	trainingLabels, validationLabels, testLabels = labels
	
	# Save each dataset / label set separately
	with open(pickleFile, 'wb') as f:
		save = {
			'trainDataset': trainingData,
			'trainLabels': trainingLabels,
			'validDataset': validationData,
			'validLabels': validationLabels,
			'testDataset': testData,
			'testLabels': testLabels,
			}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	
	# Show final pickle file size
	print 'Compressed pickle size: %s' % os.stat(pickleFile).st_size
```


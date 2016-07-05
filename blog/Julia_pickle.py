import os
import numpy as np
from scipy import ndimage
import csv
import pickle

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
            imageData = imageData.mean(axis=2)      # Flatten image into a 2D greyscale image
        imageData = imageData - (pixelDepth / 2)    # Centre the colour values around zero
        imageData = imageData / (pixelDepth / 2)    # Reduce the standard deviation to 1

        # Put image into the numpy array
        dataset[imageNum, :, :] = imageData

    # Show dataset stats
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset)

    return dataset


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
    if   ord('0') <= label <= ord('9'):     labelId = label - ord('0')
    elif ord('A') <= label <= ord('Z'):     labelId = label - ord('A') + 10
    elif ord('a') <= label <= ord('z'):     labelId = label - ord('a') + 36

    return labelId


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
            'testLabels': testLabels
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

    # Show final pickle file size
    print 'Compressed pickle size: %s' % os.stat(pickleFile).st_size


# Load, split and save datasets
pickleFiles()


##############################
#                            #
#   SANITY CHECK FUNCTIONS   #
#                            #
##############################

import matplotlib.pyplot as plt

def showImage(image):
	plt.imshow(image)
	plt.show()

def checkImages():
	images = getImageData()
	for i in xrange(5):
		showImage(images[i])

def getLabel(labelId):
    if    0 <= labelId <= 9:        label = labelId + ord('0')
    elif 10 <= labelId <= 35:       label = labelId + ord('A') - 10
    elif 36 <= labelId <= 61:       label = labelId + ord('a') - 36
    return chr(int(label))

def checkSavedData():
	with open('julia.pickle', 'rb') as f:
		save = pickle.load(f)
		trainDataset = save['trainDataset']
		trainLabels = save['trainLabels']
		validDataset = save['validDataset']
		validLabels = save['validLabels']
		testDataset = save['testDataset']
		testLabels = save['testLabels']
		del save  # Hint to help garbage collection free up memory

	for i in xrange(3):
		print 'Training label: %s' % getLabel(trainLabels[i])
		showImage(trainDataset[i])
		print 'Validation label: %s' % getLabel(validLabels[i])
		showImage(validDataset[i])
		print 'Test label: %s' % getLabel(testLabels[i])
		showImage(testDataset[i])

# Uncomment these to run sanity checks
##checkImages()
##checkSavedData()

import numpy as np
import tensorflow as tf
import csv
from scipy import ndimage
import os
import random
import pickle

def getLabel(x):
	# This gets the 'bin' for each character (0-9, then A-Z, then a-z), with 61 in total
	x = ord(x)
	if 48 <= x <= 57:		x -= 48
	elif 65 <= x <= 90:		x -= 55
	elif 97 <= x <= 122:	x -= 61
	return x

def getLabels():
	labels = []
	with open('trainLabels.csv') as f:
		reader = csv.DictReader(f)
		for line in reader:
			labels.append(line['Class'])
	labels = [getLabel(x) for x in labels]
	return labels

def getImageData():
	imageSize = 20
	pixelDepth = 255.0  # Number of levels per pixel.
	folder = 'trainResized'
	imageFiles = os.listdir(folder)
	dataset = np.ndarray(shape=(len(imageFiles), imageSize, imageSize), dtype=np.float32)
	
	numImages = 0
	for imageNum, image in enumerate(imageFiles):
		imageFile = os.path.join(folder, image)
		try:
			imageData = (ndimage.imread(imageFile).astype(float) - pixelDepth / 2) / pixelDepth
			if len(imageData.shape) == 3:
				imageData = imageData.mean(axis=2)
			dataset[numImages, :, :] = imageData
			numImages += 1
		except IOError as e:
			print 'Could not read:', imageFile, ':', e, '- it\'s ok, skipping.'
    
	dataset = dataset[0:numImages, :, :]
	print 'Full dataset tensor:', dataset.shape
	print 'Mean:', np.mean(dataset)
	print 'Standard deviation:', np.std(dataset)
	return dataset

def shuffleDataAndLabels(data, labels):
	zipped = zip(data, labels)
	random.shuffle(zipped)
	data, labels = zip(*zipped)
	return data, labels

def splitData(data, labels, validationSize, testSize):
	testData       = np.asarray(data[:testSize], dtype=np.float32)
	validationData = np.asarray(data[testSize:testSize+validationSize], dtype=np.float32)
	trainingData   = np.asarray(data[testSize+validationSize:], dtype=np.float32)
	data = [trainingData, validationData, testData]
	testLabels       = np.asarray(labels[:testSize], dtype=np.float32)
	validationLabels = np.asarray(labels[testSize:testSize+validationSize], dtype=np.float32)
	trainingLabels   = np.asarray(labels[testSize+validationSize:], dtype=np.float32)
	labels = [trainingLabels, validationLabels, testLabels]
	return data, labels

def saveData(pickleFile, (trainingData, validationData, testData), (trainingLabels, validationLabels, testLabels)):
	try:
		f = open(pickleFile, 'wb')
		save =
		{
			'train_dataset': trainingData,
			'train_labels': trainingLabels,
			'valid_dataset': validationData,
			'valid_labels': validationLabels,
			'test_dataset': testData,
			'test_labels': testLabels,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickleFile, ':', e)
		raise
	statinfo = os.stat(pickleFile)
	print 'Compressed pickle size:', statinfo.st_size

def pickleFiles():
	imageData = getImageData()
	labels = getLabels()
	
	imageData, labels = shuffleDataAndLabels(imageData, labels)

	validationSize = 1000
	testSize = 1000
	## The rest will go to training (totalSize = 6283)

	imageData, labels = splitData(imageData, labels, testSize, validationSize)
	
	saveData('julia.pickle', imageData, labels)

pickleFiles()	

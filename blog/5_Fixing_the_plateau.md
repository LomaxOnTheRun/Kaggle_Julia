#Fixing the plateau

###Overview

We're going to look at one method of fixing the plateau which occures at the start of the learning process, where the network only (and always) guesses the label 'A' for every image.

###Equalising the data

Since the problem arises from there being many more of some labels than others (most notably, 'A's), we're going to give our network batches compromised of one of each type of label. This way, the network won't be inclined to just guess the most common label, instead causing it to look for other strategies for identifying the images. We'll do this by sorting our images into 'bins', with each bin corresponing to one label.

```python
def putImagesInBins(images, labels):
	imageBins = [[] for i in xrange(62)]
	for index, image in enumerate(images):
		label = np.argmax(labels[index])
		imageBins[label].append(image)
	return imageBins

imageBins = putImagesInBins(trainDataset, trainLabels)
```

We now need to create the batches that our network will train with. The first batch will be created by choosing the first image in each bin, the second batch will be created by choosing the second image in each bin, and so on and so forth. Once a bin has run out of images, we simply select the first image from that bin instead, the the second image, and so on.

```python
def getBatchData(step):
	'''Get one of each type of image'''
	# Images
	batchImages = np.ndarray(shape=(62, imageSize * imageSize), dtype=np.float32)
	for index, imageBin in enumerate(imageBins[:62]):
		showStep = step
		while showStep >= len(imageBin):
			showStep -= len(imageBin)
		image = imageBin[showStep]
		batchImages[index, :] = image
	# Labels
	labels = np.arange(numLabels)
	batchLabels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return batchImages, batchLabels
```

We now just have a couple of things to take care of before our script can run again.

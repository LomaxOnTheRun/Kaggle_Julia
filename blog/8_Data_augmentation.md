# Data augmentation

### Overview

We're going to increase the number of available training data by tweaking images slightly. This will help the network by giving it 'more examples' to work with, as well as stopping it from being able to overfit the training data.

### The code

The main differene to our code will come from a function we add which will take the batch of training images we choose, and add a number of possible filters to each one.

```python
from scipy import ndimage
from random import randint, choice

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
```

We then plug this in to our ```getBatchData()``` function.

```python
def getBatchData(step):
	'''Get one of each type of image'''
	# Images
	batchImages = np.ndarray(shape=(62, imageSize, imageSize, numChannels), dtype=np.float32)
	for index, imageBin in enumerate(imageBins[:62]):
		showStep = step
		while showStep >= len(imageBin):
			showStep -= len(imageBin)
		image = imageBin[showStep]
		batchImages[index, :, :, :] = image
	batchImages = tweakImages(batchImages)
	# Labels
	labels = np.arange(numLabels)
	batchLabels = (np.arange(numLabels) == labels[:, None]).astype(np.float32)
	return batchImages, batchLabels
```

We can then run our script and get the following graph:

![Graph 1](/images/Julia_8_blog_1.png)

We can see that the training and validation accuracies are much closer than before. This happens for two reasons:

 1. The training data can't overfit, as we're not showing our network exactly the same images any more.
 2. Due to the increase intraining data, our network is better at generalising with images it's never seen before.


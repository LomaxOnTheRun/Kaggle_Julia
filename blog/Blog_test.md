#Getting the training data

###What?

In this blog entry we will be downloading, pre-processing and storing the training images for our network to use.

###How?

Kaggle handily offers a 'Data' page for each of its competitions. For this particular competition, there are two sets of training data: one set of raw pictures, each with varying heights and widths, and one that they've resized to 20x20 pixels. As I'm keeping things small, the 20x20 images will do nicely for now. We can always play around with larger images once we've got a network we know works.

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

###Why?

This is why.

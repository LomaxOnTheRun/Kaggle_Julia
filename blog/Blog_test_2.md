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

This is obviously not yet code that we can run - we still need to define all the functions we call. Nonetheless, it's still useful insofar as it gives us a clear idea of what we need to do. Just before we get down to creating all of the required functions, we'll take a quick look at why we need validation and test datasets, as well as our training dataset.

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


1. We centred the values of the pixels around zero.
 
We've done this to better 'define' the problem. Gradient descent methods (which we'll be using) have a much easier time working with well-defined problems, which will manifest itself as a faster learning rate and more accurate results. (MORE INFO AND PICS)

Centred problem | Uncentred problem
----------------|--------------------

![Pic1](/images/GradientDescent_centred_small.png) | ![Pic2](images/GradientDescent_uncentred_small.png)

2. We restricted the range of values the pixels can take to between -1 and 1.

This will help us to reduce floating point errors. When dealing with floating point numbers (e.g. Python's ```float``` numbers), small rounding errors occur as the processor is not able to hold infinitely long decimal places in memory. This only really starts becomeing a problem when trying to calculate tiny difference between very large numbers and, while 255 may not look very large, it becomes problematic when we use large networks with very large matrices. Remember, we'll be trying to squeeze out every possible bit of accuracy from our network, so even a fraction of a percent will make a difference.

###Getting the labels

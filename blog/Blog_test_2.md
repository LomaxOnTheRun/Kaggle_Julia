#Getting the training data

###Overview

In this session, we will download the training data from Kaggle, pre-process it, and then save it to a Pickle file using Python.

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

This is obviously not yet code that we can run - we still need to define all the functions we call. Nonetheless, it's still useful insofar as it gives us a clear idea of what we need to do. Just before we get down to creating all of the required functions, we'll take a quick look at why we need validation and test datasets, as well as our training set.

###Why do we need validation and test datasets?

The reason for having a test dataset is very simple: we need to know how good our network is at identifying images *it has never seen before*.

The reason for having a validation dataset is slightly more subtle. Whilst we won't ever show the network any of the images from the validation dataset either, we will be using it to monitor the progress of our network, and to change the hyper-parameters of the network accordingly. In this way, some of the information about the validation dataset will bleed through to the network. To stop the same thing from happening to our test dataset, we will only ever check how accurate our network is against it once, after it has completely finished training.

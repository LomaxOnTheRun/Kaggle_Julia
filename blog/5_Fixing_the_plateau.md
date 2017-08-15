# Fixing the plateau

### Overview

We're going to look at one method of fixing the plateau which occures at the start of the learning process, where the network only (and always) guesses the label 'A' for every image.

### Equalising the data

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

We now just have a couple of things to take care of before our script can run again. The first is that we need to pass in the current step number into our ```getBatchData()``` function.

```python
			batchDataset, batchLabels = getBatchData(step)
```

The second is that we need to change the batch size to be the same as the number of labels we have.

```python
batchSize = numLabels
```

When we run our new script, we get a graph that looks like this:

![Graph 1](/images/Julia_5_blog_1.png)

This is obviously much worse that before, but with the benefit of not having the validation accuracy stuck at 6.2%. This will help us overcome a major obstacle later on, but if you want to ignore this step for now and continue the tutorial without it, you absolutely can. For the rest of us, let's look at getting our accuracy up again.

### Getting back up to speed

We're now going to try to adjust the learning rate hyperparameter to optimise the new setup of the network. To save us time, we're going to assume that the value we have is roughly correct, so we're only going to try doubling and halving them, then fine turning them a little more.

![Graph 2](/images/Julia_5_blog_2.png)

We get a very significant improvement by using 0.4 as the learning rate. We can try to fine tune this a little more.

![Graph 3](/images/Julia_5_blog_3.png)

Nope, it looks like 0.4 is indeed our best value. We can now run this for 30,000 steps to see how our network deals with this in the long run.

![Graph 4](/images/Julia_5_blog_4.png)

### Full script

You can get the full script for this post [here](/blog/Julia_5.py).

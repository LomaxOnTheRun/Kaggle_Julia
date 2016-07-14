#Fixing the plateau

###Overview

We're going to look at one method of fixing the plateau which occures at the start of the learning process, where the network only (and always) guesses the label 'A' for every image.

###Equalising the data

Since the problem arises from there being many more of some labels than others (most notably, 'A's), we're going to give our network batches compromised of one of each type of label. This way, the network won't be inclined to just guess the most common label, instead causing it to look for other strategies for identifying the images. We'll do this by sorting our images into 'bins', with each bin corresponing to one label.

```python

```

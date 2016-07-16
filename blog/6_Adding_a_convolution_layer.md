#Adding a convolution layer

###Overview

We're going to be adding a convolution layer to our network now. This is going to allow the network to create several 'filters' which it will apply to each of the images, and it will be the filtered images which will be put through the 2 hidden layers. Convolution layers are substantially lighter than their hidden layer counterparts, as only the kernels get updated, not every connection weight for every node.

###Adding the layer


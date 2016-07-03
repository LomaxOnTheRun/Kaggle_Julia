#Building our first neural network

###Overview

In this post, we'll be creating our first and most basic neural network, using TensorFlow. This is Google's contribution to the field of open-sourced neural network libraries. It uses a C++ backend, which allows for faster computations, although it can make it trickier to use to begin with, and more difficult to implement and tweak your own training algorithms later on. However, for our purposes it will do very well.

I should mention now that there is a lot going on in this script. As with the last post, I'll go through and explain each part as I go, but there are quite a few aspect of this that we need to look at to create a coherent and useful script. There is a link at the bottom to the finished script, so feel free to play around with it first, then come back and read up on any parts that don't make sense.

###A first look at our network code

This script will be roughly broken down into 4 parts:

1. Loading the pickled data
2. Setting up the network
3. Running the network
4. Plotting the network accuracy

So without further ado:

###Loading the pickled data


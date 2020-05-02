# Deep Embedded Clustering

Deep Embedded Clustering is an unsupervised machine learning algorithm to cluster unlabeled data. This model pretrains an autoencoder first, and then creates and trains a DEC model consisting of an Encoder and a Custom layer. The weights of the encoder are initialized from the pretrained autoencoder and custom layer's weights are initialized from the cluster centers found from running kMeans on the pretrained encoder's output.  DEC model uses KL-Divergence as the loss function to measure the matching between different distributions.

Improved DEC model with convolutional autoencoder is implemented on GOES satellite data to find cloud clustering.

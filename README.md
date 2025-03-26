**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 2.**

Q1: Implementing a Basic Autoencoder

Task: Autoencoders learn to reconstruct input data by encoding it into a lower-dimensional space. You will build a fully connected autoencoder and evaluate its performance on image reconstruction.

1.Data Loading and Preprocessing:
  The MNIST dataset is loaded.
  Pixel values are normalized to the range [0, 1].
  Images are flattened into 784-dimensional vectors.
  
2.Autoencoder Architecture:
  The encoder compresses the input into a lower-dimensional "latent" space.
  The decoder reconstructs the input from the latent representation.
  The sigmoid activation function is used on the output layer because the loss function is 
  binary crossentropy.
  The relu activation function is used on the hidden layers.
  
3.Compilation and Training:
  The autoencoder is compiled using the Adam optimizer and binary cross-entropy loss.
  The model is trained for 50 epochs.

4.Visualization:
  The original and reconstructed images are plotted side by side for visual comparison.

5.Latent Dimension Analysis:
  The latent dimension is varied (16 and 64).
  The autoencoder is retrained and the reconstruction quality is observed.

 Observations:
  A smaller latent dimension (16) results in more information loss, leading to blurrier and less 
  detailed reconstructions. The model is forced to compress the image information more severely.
  A larger latent dimension (64) allows the autoencoder to retain more information, resulting in 
  sharper and more detailed reconstructions. The model has more capacity to represent the input.
  The initial latent dimension of 32 provides a good balance between compression and 
  reconstruction quality.

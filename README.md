**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 3.**

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
  sharper and more detailed reconstructions.
  
  The model has more capacity to represent the input.
  
  The initial latent dimension of 32 provides a good balance between compression and 
  reconstruction quality.

  Q2: Implementing a Denoising Autoencoder

Task: Denoising autoencoders can reconstruct clean data from noisy inputs. You will train a model to remove noise from images.

1.Noise Addition:
  np.random.normal() is used to generate Gaussian noise with a mean of 0 and a standard deviation of 0.5.
  
  This noise is added to the input images (x_train_noisy, x_test_noisy).
  
  np.clip() ensures that the noisy pixel values remain within the valid range [0, 1].
  
2.Denoising Autoencoder Training:
  Crucially, the autoencoder is trained with the noisy images as input (x_train_noisy) and the clean images as the target output (x_train). This is what forces the model to learn to remove noise.

3.Visualization:
  The visualization now includes three rows: original images, noisy images, and reconstructed images. This allows for a direct visual comparison of the denoising effect.

4.Comparison:
  A qualitative comparison between the basic and denoising autoencoders is made. The denoising autoencoder clearly produces cleaner reconstructions from noisy inputs.

5.Real-World Scenario:
  Medical Imaging: Denoising autoencoders are highly useful in medical imaging to improve the quality of images corrupted by noise, such as those from CT or MRI scans. This can lead to more accurate diagnoses.
  Other applications include:
    Security: Denoising surveillance footage or enhancing low-light images.
    Astronomy: Removing noise from telescope images.
    Audio processing: Removing static or background noise from audio signals.

Q3: Implementing an RNN for Text Generation

Task: Recurrent Neural Networks (RNNs) can generate sequences of text. You will train an LSTM-based RNN to predict the next character in a given text dataset.

1.Data Loading and Preprocessing:
  Loads the Shakespeare text file.
 
  Converts the text to lowercase.
  
  Creates character-to-index and index-to-character mappings.
  
  Generates sequences of maxlen characters and their corresponding next characters.
 
  One-hot encodes the sequences and next characters.

2.RNN Model:
  An LSTM layer with 128 units is used to capture long-range dependencies in the text.

  A dense output layer with a softmax activation function predicts the probability distribution of the next character.

3.Text Generation:
  The sample() function applies temperature scaling to the predicted probabilities and samples the next character.

  The generate_text() function generates text by repeatedly predicting and sampling characters.

  Text generation is performed at different temperatures (0.2, 1.0, and 1.2) to demonstrate the effect of temperature scaling.

4.Temperature Scaling:
  Temperature scaling is crucial for controlling the randomness of text generation.

  A lower temperature makes the model more confident, leading to less diverse but more coherent text.

  A higher temperature increases randomness, resulting in more diverse but potentially nonsensical text.

  A temperature of zero would cause the model to always choose the most likely character.

Key Improvements:

  Clearer comments and explanations.

  Demonstrates text generation at different temperatures.

  Explains the role of temperature scaling in detail.

  Includes the necessary import statements.

  Uses a better text source.

  Improves the clarity of the code and the printed output.

  Adds a comment about the effect of 0 temperature.

  Q4: Sentiment Classification Using RNN

Task: Sentiment analysis determines if a given text expresses a positive or negative emotion. You will train an LSTM-based sentiment classifier using the IMDB dataset.

1.Data Loading:
  Loads the IMDB dataset using tf.keras.datasets.imdb.load_data().

  num_words=10000 limits the vocabulary size, which helps reduce memory usage and training time.

2.Preprocessing:
  sequence.pad_sequences() is used to pad or truncate reviews to a fixed length (max_review_length). This is essential for feeding sequences into the LSTM.

3.LSTM Model:
  An Embedding layer converts word indices into dense vectors.
  An LSTM layer processes the sequences and captures long-range dependencies.
  A Dense output layer with a sigmoid activation function predicts the sentiment (positive or negative).
  The model is compiled with binary cross-entropy loss and the Adam optimizer.

4.Evaluation:
  model.predict() generates probability predictions.

  The probabilities are converted into binary predictions (0 or 1).

  confusion_matrix() and classification_report() from scikit-learn are used to evaluate the model's performance.

  The confusion matrix is also visualized using a heatmap.

5.Precision-Recall Tradeoff:
  The code includes an explanation of the precision-recall tradeoff and its importance in sentiment classification.

  It explains that depending on the application, either precision or recall could be more important.

  It gives real world examples of the importance of each metric.

  This is a very important concept in machine learning, and especially in binary classification.

  The code now includes a visualization of the confusion matrix.

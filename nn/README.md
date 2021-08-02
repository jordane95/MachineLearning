A naïve LSTM implementation with Numpy, model performance tested on

- [x] image classification (MNIST)
- [x] sentiment analysis (IMDB)
- [x] language modelling (Penn Tree)

##  Experiment results

### Image classification

We use the MNIST dataset to train our model for image classification (10 classes). The overall model architecture is LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss.

Due to memory limitation, we only use the first 10k images in the training set to train our model, but all images in the test set to evaluate the model. The image data are firstly normalized to range [0, 1] for better convergence. The image matrix is regarded as a sequence of vectors. The batch size is set to be 64. 

The TensorFlow model weights are initialized with Xavier initializer by default. It's trained with SGD optimizer for 20 epochs. The learning rate is 1.0.

The weights of my model are initialized with the Normal distribution, the bias are all initialized to 0. My model is trained with SGD for 100 epochs. The initial learning rate is 1.0, and the decay rate is 0.2 per epoch, that is, the learning at epoch n is
$$
learning\_rate_n = \frac{1}{1+0.2n}
$$
The experiment results are as follows

| Model      | test accuracy | train accuracy | train loss |
| ---------- | ------------- | -------------- | ---------- |
| TensorFlow | 96.4          | 98.4           | 0.05       |
| Mine       | 93.4          | 97.2           | 0.13       |

### Sentiment analysis

We trained our models on the IMDB dataset for sentiment analysis (binary text classification). The model architecture is Embedding + LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss. The embedding lookup matrix is initialized with the 50-dimension Glove word vectors.

My model is trained with the SGD optimizer early stopped at 50 epochs where the gradients begin exploding. Due to memory limitation, it's trained only with the first 1k sentences, and tested with the first 500 sentences. The hyperparameters settings are as follows:

* random seed = 42

* batch size = 32

* hidden size = 32
* initial learning rate = 1.0

* learning rate decay rate = 0.2

* weight initialization: W ~ 0.01* N(0, 1) for LSTM layer, W ~ N(0, 1) for linear layer

* gradient threshold = 5

The TensorFlow model is trained with the first 5k sentences. The optimizer is SGD with learning rate 1. We train the model for 50 epochs, with batch size = 32, hidden size = 32.

The results are shown as follows (w/o denote without updating embedding layer, whereas w denote update)

| Model            | test accuracy | train accuracy | train loss |
| ---------------- | ------------- | -------------- | ---------- |
| mine (w/o)       | 73.0          | 81.7           | 0.460      |
| mine (w)         | 77.4          | 86.6           | 0.413      |
| TensorFlow (w/o) | 78.6          | 92.7           | 0.196      |
| TensorFlow (w)   | 81.8          | 100.0          | 0.004      |

### Language modelling

We use the Penn Tree Bank dataset to train our LSTM language model. The model architecture is Embedding layer + LSTM + Linear Layer + Softmax Layer + Cross Entropy loss. The word embedding matrix is initialized by 50 dimensional GloVe vectors. The hidden size of recurrent layer is set to be 32. The linear layer share parameters across time. The word embeddings are fixed during training.

The hyperparameter settings:

* vocab size = 10000
* embedding size = 50
* hidden size = 32
* batch size = 128
* time step = 60
* initial learning rate = 1.0

My model in Numpy is trained with SGD optimizer with learning rate = 1.0 for almost 7 hours. The TensorFlow model is trained with SGD optimizer for  almost 2 hours. We also trained the TensorFlow model with Adam, which turns out to converge faster.  The experiment results are shown as follows, where the perplexity is calculated as 
$$
perplexity = e^{CrossEntropyLoss}
$$
| Model                        | test loss / perplexity | train loss / perplexity |
| ---------------------------- | ---------------------- | ----------------------- |
| Mine (SGD 50 epochs)         | 6.15 / 468.71          | 6.19 / 487.85           |
| TensorFlow (SGD 50 epochs)   | 6.23 / 507.76          | 6.28 / 533.79           |
| TensorFlow (Adam 50 epochs)  | 5.85 / 347.23          | 5.86 / 350.72           |
| TensorFlow (Adam 100 epochs) | 5.71 / 301.87          | 5.65 / 284.29           |

A naïve LSTM implementation, performance tested on 

- [x] image classification (MNIST)
- [x] sentiment analysis (IMDB)
- [x] language modelling (Penn Tree)

[^_^]:sequence labelling (Penn Tree)

##  Experiment results

### Image classification

We use the MNIST dataset to train our model for image classification (10 classes). The overall model architecture is Embedding + LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss.

The image data are firstly normalized to be in range [0, 1] for better convergence. The batch size is 64. 

The TensorFlow model parameters are initialized with Xavier initializer by default. It's trained with SGD optimizer for 20 epochs. The learning rate is 1.0 for 20 epochs.

The weights of my model are initialized with the Normal distribution, its bias are all initially set to be 0. My model is trained with SGD for 100 epochs. The initial learning rate is 1.0, and the decay rate is 0.2 per epoch, that is, the learning at epoch n is
$$
learning\_rate_n = \frac{1}{1+0.2n}
$$
The experiment results are as follows

| Model      | test accuracy | train accuracy | train loss |
| ---------- | ------------- | -------------- | ---------- |
| TensorFlow | 96.4          | 98.4           | 0.05       |
| Mine       | 93.4          | 97.2           | 0.13       |

### Sentiment analysis

We use the IMDB dataset for sentiment analysis (binary text classification). The model architecture is Embedding + LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss. The embedding lookup matrix is initialized with the Glove word vectors, without updating during training.

My model is trained with the SGD optimizer early stopped at 50 epochs where the gradients begin exploding. The hyperparameters settings are as follows:

* random seed = 42

* batch size = 32

* hidden size = 32
* initial learning rate = 1.0

* learning rate decay rate = 0.2

* weight initialization: W ~ 0.01* N(0, 1) for LSTM layer, W ~ N(0, 1) for linear layer

* gradient threshold = 5

For comparison, the model written in TensorFlow is trained at learning rate 1 for 50 epochs, with batch size = 32, hidden size = 32.

| Model      | test accuracy | train accuracy | train loss |
| ---------- | ------------- | -------------- | ---------- |
| TensorFlow | 78.6          | 92.7           | 0.196      |
| mine       | 73.0          | 81.7           | 0.460      |

### Language modelling

We use the Penn Tree Bank dataset to train our LSTM language model. The model architecture is Embedding layer + LSTM + Linear Layer + Softmax Layer + Cross Entropy loss. The word embedding matrix is initialized by GloVe, with embedding size = 50. The hidden size of recurrent layer is set to be 32. The linear layer share parameters across time. 

The hyperparameters:

* vocab size = 10000
* embedding size = 50
* hidden size = 32
* batch size = 128
* time step = 60
* initial learning rate = 1.0

The TensorFlow model is trained with SGD optimizer for 50 epochs, almost 2 hours. My model is trained with SGD optimizer with learning rate = 1.0 for almost 7 hours. With the definition, the perplexity can be calculated as 
$$
perplexity = e^{CrossEntropyLoss}
$$
The experiment results are shown as follows

| Model                        | test loss / perplexity | train loss / perplexity |
| ---------------------------- | ---------------------- | ----------------------- |
| Mine (SGD 50 epochs)         | 6.15 / 468.71          | 6.19 / 487.85           |
| TensorFlow (SGD 50 epochs)   | 6.23 / 507.76          | 6.28 / 533.79           |
| TensorFlow (Adam 50 epochs)  | 5.85 / 347.23          | 5.86 / 350.72           |
| TensorFlow (Adam 100 epochs) | 5.71 / 301.87          | 5.65 / 284.29           |




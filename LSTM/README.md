A naïve LSTM implementation, performance tested on 

- [x] image classification (MNIST)
- [x] sentiment analysis (IMDB)
- [ ] language modelling (Penn Tree)
- [ ] sequence labelling (Penn Tree)

##  Experiment results

### Image classification

We use the MNIST dataset to train our model for image classification. The overall model architecture is Embedding + LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss. The models are trained using the SGD optimizer, with the batch size of 64. 

The TensorFlow model is trained with learning rate of 1.0 for 20 epochs, whereas my model is trained with initial learning rate of 1.0 and decay rate of 0.2 for 100 epochs.

| Model      | test accuracy | train accuracy | loss  |
| ---------- | ------------- | -------------- | ----- |
| TensorFlow | 96.8          | 99.4           | 0.023 |
| my model   | 93.4          | 97.2           | 0.133 |

### Sentiment analysis

We use the IMDB dataset for sentiment analysis (binary text classification). The model architecture is Embedding + LSTM + Average Pooling + Linear + Softmax + Cross Entropy Loss. The embedding lookup matrix is initialized with the Glove word vectors, without updating during training.

My model is trained with the SGD optimizer early stopped at 50 epochs where the gradients begin exploding. The hyperparameters settings are as follows:

* random seed = 42

* batch size=32

* hidden size=32
* initial learning rate = 1.0

* learning rate decay rate = 0.2

* weight initialization: W ~ 0.1* N(0, 1) for LSTM layer, W ~ N(0, 1) for linear layer

* gradient threshold = 5

For comparison, the model written in TensorFlow is trained at learning rate 1 for 20 epochs, with batch size=32, hidden size=32.

| Model      | test accuracy | train accuracy | loss  |
| ---------- | ------------- | -------------- | ----- |
| TensorFlow | 73.8          | 87.8           | 0.301 |
| my model   | 73.0          | 81.7           | 0.460 |

### Language modelling

For the language modelling task, we choose the Penn Tree Bank dataset. The model architecture is the same as the preceding model for sentiment analysis. 

| Model      | test perplexity | train perplexity |
| ---------- | --------------- | ---------------- |
| tensorflow |                 |                  |
| my model   |                 |                  |


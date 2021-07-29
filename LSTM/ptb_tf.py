import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from glove import get_glove_vec
from ptb_loader import read_data, make_batch, get_word_index


TRAIN_DIR = 'data/ptb.train.txt'
EVAL_DATA = 'data/ptb.valid.txt'
TEST_DATA = 'data/ptb.test.txt'

MAX_EPOCH = 50

BATCH_SIZE = 128
TIME_STEP = 60
VOCAB_SIZE = 10000
EMBED_SIZE = 50
HIDDEN_SIZE = 32

# loading training data
data_batches, label_batches = make_batch(read_data('data/ptb.train'), batch_size=BATCH_SIZE, time_step=TIME_STEP)
word_index = get_word_index()


We = get_glove_vec(save_path='glove.6B.50d.txt', word_index=word_index, word_dim=EMBED_SIZE)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, embeddings_initializer=tf.keras.initializers.Constant(We)))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add()

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from glove import get_glove_vec
from ptb.ptb_loader import read_data, make_batch, get_word_index

MAX_EPOCH = 100

BATCH_SIZE = 128
TIME_STEP = 60
VOCAB_SIZE = 10000
EMBED_SIZE = 50
HIDDEN_SIZE = 32

# loading training data
data_batches, label_batches = make_batch(read_data('ptb/ptb.train'), batch_size=BATCH_SIZE, time_step=TIME_STEP)
word_index = get_word_index(vocab_dir='ptb/ptb.vocab')
# load test data
test_data_batches, test_label_batches = make_batch(read_data('ptb/ptb.test'), batch_size=BATCH_SIZE, time_step=TIME_STEP)


We = get_glove_vec(save_path='glove/glove.6B.50d.txt', word_index=word_index, word_dim=EMBED_SIZE)


class PTBLM(Model):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(PTBLM, self).__init__()
        self.embed = Embedding(vocab_size, embed_size, embeddings_initializer=tf.keras.initializers.Constant(We), trainable=False)
        self.rec = LSTM(hidden_size, return_sequences=True)
        self.linear = Dense(vocab_size)

    def call(self, xs):
        xs = self.embed(xs)
        xs = self.rec(xs)
        xs = self.linear(xs)
        xs = tf.reshape(xs, (-1, xs.shape[-1]))
        return xs


model = PTBLM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(data, label):
    with tf.GradientTape() as tape:
        pred = model(data)
        loss = loss_object(tf.reshape(label, (-1, label.shape[-1])), pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(data, label):
    pred = model(data)
    loss = loss_object(tf.reshape(label, (-1, label.shape[-1])), pred)

    test_loss(loss)


def test():
    data = data_batches[0]
    label = label_batches[0]
    for epoch in range(100):
        train_loss.reset_states()
        train_step(data, label)
        print(
            f'Epoch {epoch+1}, '
            f'Loss {train_loss.result()}'
        )


def train():
    for epoch in range(MAX_EPOCH):
        train_loss.reset_states()
        test_loss.reset_states()

        for data, label in zip(data_batches, label_batches):
            train_step(data, label)

        for test_data, test_label in zip(test_data_batches, test_label_batches):
            test_step(test_data, test_label)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}'
        )


if __name__ == '__main__':
    train()

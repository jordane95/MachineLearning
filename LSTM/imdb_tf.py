import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, GlobalAvgPool1D, Dense
from tensorflow.keras import Sequential
from glove import get_glove_vec
from data import load_imdb

vocab_size = 10000
embed_size = 50
hidden_size = 32
output_size = 2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

(x_train, y_train), (x_test, y_test), word_index = load_imdb()
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:500], y_test[:500]

# word vector look up matrix initialization
We = get_glove_vec(save_path='./glove.6B.50d.txt', word_index=word_index, word_dim=embed_size)[:10000]

model = Sequential()
model.add(Embedding(vocab_size, embed_size, embeddings_initializer=tf.keras.initializers.Constant(We), trainable=False))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(GlobalAvgPool1D())
model.add(Dense(output_size, activation='softmax'))

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=1),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=64)

print('test loss: ', score[0])
print('test accuracy: ', score[1])

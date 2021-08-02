import tensorflow as tf
from tensorflow.keras.layers import LSTM, GlobalAvgPool1D, Dense
from tensorflow.keras import Sequential


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = x_train[:10000], y_train[:10000]
x_train, x_test = x_train / 255.0, x_test / 255.0


hidden_dim = 50
output_dim = 10

model = Sequential()
model.add(LSTM(hidden_dim, return_sequences=True))
model.add(GlobalAvgPool1D())
model.add(Dense(output_dim, activation='softmax'))

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=1),
              metrics='accuracy')

model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))

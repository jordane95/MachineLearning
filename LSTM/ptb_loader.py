import numpy as np


def read_data(train_dir):
    with open(train_dir, 'r', encoding='utf-8') as f:
        id_str = ' '.join([line.strip() for line in f])
        ids = [int(w) for w in id_str.split()]
    return ids


def make_batch(ids, batch_size, time_step):
    num_batches = (len(ids)-1) // (batch_size*time_step)
    data = np.array(ids[:num_batches*batch_size*time_step]).reshape((batch_size, -1))
    data_batches = np.split(data, num_batches, axis=1)
    label = np.array(ids[1:num_batches * batch_size * time_step + 1]).reshape((batch_size, -1))
    label_batches = np.split(label, num_batches, axis=1)
    return data_batches, label_batches


def get_word_index(vocab_dir='data/ptb.vocab'):
    with open(vocab_dir, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]
    word_index = {w: idx for w, idx in zip(words, range(len(words)))}
    return word_index

import numpy as np


def get_glove_vec(save_path, word_index, word_dim):
    word_embeddings = np.zeros((len(word_index), word_dim))
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_index:
                word_embeddings[word_index[word]] = np.asarray(values[1:], dtype='float32')
    return word_embeddings

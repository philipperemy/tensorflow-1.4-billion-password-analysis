import numpy as np
from keras.layers import Concatenate
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Input, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

vocab_size = 500
max_length = 20
batch_size = 1


def get_model():
    i1 = Input(batch_shape=(batch_size, max_length))
    i2 = Input(batch_shape=(batch_size, max_length))
    share_embedding = Embedding(vocab_size, 8, input_length=max_length)
    m1 = share_embedding(i1)
    m2 = share_embedding(i2)
    m1 = LSTM(32)(m1)
    m2 = LSTM(32)(m2)
    m1 = Dense(32, activation='relu')(m1)
    m2 = Dense(32, activation='relu')(m2)
    o = Concatenate(axis=1)([m1, m2])
    o = Dense(1, activation='sigmoid')(o)
    return Model(inputs=[i1, i2], outputs=[o])


def infinite_read(file, mode, encoding):
    r_fp = open(file='1.csv', mode='r', encoding='utf8')
    return read_in_chunks(r_fp)


def read_in_chunks(file_object):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


def run():
    m = get_model()
    m.compile(optimizer='adam', loss='binary_crossentropy')

    while True:
        for piece in infinite_read(file='1.csv', mode='r', encoding='utf8'):
            print(piece)
            ed, pair_1, pair_2 = piece.strip().split('|||')
            print('-')
            pair_1_encoding = sum([one_hot(d, vocab_size) for d in list(pair_1.strip())], [])
            pair_2_encoding = sum([one_hot(d, vocab_size) for d in list(pair_2.strip())], [])
            padded_1 = pad_sequences([pair_1_encoding], maxlen=max_length, padding='post')
            padded_2 = pad_sequences([pair_2_encoding], maxlen=max_length, padding='post')
            targets = np.ones(shape=1)
            m.fit(x=[padded_1, padded_2], y=targets)
            # print(m.train_on_batch(x=[padded_1, padded_2], y=[[[1.0]]]))


if __name__ == '__main__':
    run()

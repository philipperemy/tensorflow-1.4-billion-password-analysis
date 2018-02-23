# -*- coding: utf-8 -*-


import os
import random

import numpy as np
from keras import layers
from keras.models import Sequential

from constants import MAX_PASSWORD_LENGTH
from data_gen import LazyDataLoader, get_chars_and_ctable, colors

DATA_LOADER = LazyDataLoader()

INPUT_MAX_LEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

chars, c_table = get_chars_and_ctable()


def gen_large_chunk(inputs, targets, chunk_size):
    print('x.shape =', (chunk_size, MAX_PASSWORD_LENGTH, len(chars)))
    print('y.shape =', (chunk_size, MAX_PASSWORD_LENGTH, len(chars)))

    # unless we have scipy.sparse matrices, it's too big to fit in memory.
    # lets do lazy convert (only when we present the batch inside the training module).

    random_indices = np.random.choice(a=range(len(inputs)), size=chunk_size, replace=True)
    sub_inputs = inputs[random_indices]
    sub_targets = targets[random_indices]

    x = np.zeros((chunk_size, MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((chunk_size, MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)

    for i_, element in enumerate(sub_inputs):
        x[i_] = c_table.encode(element, MAX_PASSWORD_LENGTH)
    for i_, element in enumerate(sub_targets):
        y[i_] = c_table.encode(element, MAX_PASSWORD_LENGTH)

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print('Done... File is /tmp/x_y.npz')
    return x_train, y_train, x_val, y_val


if not os.path.exists('/tmp/x_y.npz'):
    raise Exception('Please run the vectorization script before.')

print('Loading data from prefetch...')
data = np.load('/tmp/x_y.npz')
inputs = data['inputs']
targets = data['targets']

print('Data:')
print(inputs.shape)
print(targets.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')


def model_1():
    m = Sequential()
    m.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(layers.RepeatVector(OUTPUT_MAX_LEN))
    for _ in range(LAYERS):
        m.add(RNN(HIDDEN_SIZE, return_sequences=True))
    m.add(layers.TimeDistributed(layers.Dense(len(chars))))
    m.add(layers.Activation('softmax'))
    return m


def model_2():
    # too big in Memory!
    m = Sequential()
    from keras.layers.core import Flatten, Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(Flatten(input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars)))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


def model_3():
    m = Sequential()
    from keras.layers.core import Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars)))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


model = model_3()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation data set.
for iteration in range(1, 200):
    x_train, y_train, x_val, y_val = gen_large_chunk(inputs, targets, chunk_size=BATCH_SIZE * 1000)
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]  # replace by x_val, y_val
        preds = model.predict_classes(rowx, verbose=0)
        q = c_table.decode(rowx[0])
        correct = c_table.decode(rowy[0])
        guess = c_table.decode(preds[0], calc_argmax=False)
        print('T', correct)
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=" ")
        else:
            print(colors.fail + '☒' + colors.close, end=" ")
        print(guess)
        print('---')

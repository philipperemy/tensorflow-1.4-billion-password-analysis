# -*- coding: utf-8 -*-


import os

import numpy as np
from keras import layers
from keras.models import Sequential
from six.moves import range

from model.data_gen import LazyDataLoader, get_chars_and_ctable, colors

DATA_LOADER = LazyDataLoader()

INPUT_MAX_LEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

chars, ctable = get_chars_and_ctable()

if not os.path.exists('x_y.npz'):
    raise Exception('Please run the vectorization script before.')

print('Loading data from prefetch...')
data = np.load('x_y.npz')
x_train = data['x_train']
x_val = data['x_val']
y_train = data['y_train']
y_val = data['y_val']

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

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
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('T', correct)
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=" ")
        else:
            print(colors.fail + '☒' + colors.close, end=" ")
        print(guess)
        print('---')

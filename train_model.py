# -*- coding: utf-8 -*-

import multiprocessing
import os

import numpy as np
from keras import layers
from keras.models import Sequential

from constants import MAX_PASSWORD_LENGTH
from data_gen import get_chars_and_ctable, colors

INPUT_MAX_LEN = MAX_PASSWORD_LENGTH
OUTPUT_MAX_LEN = MAX_PASSWORD_LENGTH
chars, c_table = get_chars_and_ctable()


def gen_large_chunk_single_thread(inputs_, targets_, chunk_size):
    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    x = np.zeros((chunk_size, MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((chunk_size, MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)

    for i_, element in enumerate(sub_inputs):
        x[i_] = c_table.encode(element, MAX_PASSWORD_LENGTH)
    for i_, element in enumerate(sub_targets):
        y[i_] = c_table.encode(element, MAX_PASSWORD_LENGTH)

    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    return x_train, y_train, x_val, y_val


def gen_large_chunk_multi_thread(inputs_, targets_, chunk_size):
    ''' This function is actually slower than gen_large_chunk_single_thread()'''

    def parallel_function(f, sequence, num_threads=None):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(processes=num_threads)
        result = pool.map(f, sequence)
        cleaned = np.array([x for x in result if x is not None])
        pool.close()
        pool.join()
        return cleaned

    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    def encode(elt):
        return c_table.encode(elt, MAX_PASSWORD_LENGTH)

    num_threads = multiprocessing.cpu_count() // 2
    x = parallel_function(encode, sub_inputs, num_threads=num_threads)
    y = parallel_function(encode, sub_targets, num_threads=num_threads)

    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

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
BATCH_SIZE = 256
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
for iteration in range(1, int(1e9)):
    x_train, y_train, x_val, y_val = gen_large_chunk_single_thread(inputs, targets, chunk_size=BATCH_SIZE * 500)
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=5,
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
        # p = model.predict(rowx, batch_size=32, verbose=0)[0]
        # p.shape (12, 82)
        # [np.random.choice(a=range(82), size=1, p=p[i, :]) for i in range(12)]
        # s = [np.random.choice(a=range(82), size=1, p=p[i, :])[0] for i in range(12)]
        # c_table.decode(s, calc_argmax=False)
        # Could sample 1000 and take the most_common()
        print('new    :', correct)
        print('former :', q)
        print('guess  :', guess, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close)
        else:
            print(colors.fail + '☒' + colors.close)
        print('---')

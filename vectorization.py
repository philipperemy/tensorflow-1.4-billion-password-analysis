from time import sleep

import numpy as np
from tqdm import tqdm

from constants import MAX_PASSWORD_LENGTH
from data_gen import LazyDataLoader, build_vocabulary, get_token_indices, get_chars_and_ctable

print('Building vocabulary...')

build_vocabulary()

print('Vectorization...')

DATA_LOADER = LazyDataLoader()

_, _, TRAINING_SIZE = DATA_LOADER.statistics()

TOKEN_INDICES = get_token_indices()

chars, c_table = get_chars_and_ctable()

inputs = []
targets = []
print('Generating data...')
while len(inputs) < TRAINING_SIZE:
    x_, y_ = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    inputs.append(x_)
    targets.append(y_)

print('x.shape =', (len(inputs), MAX_PASSWORD_LENGTH, len(chars)))
print('y.shape =', (len(inputs), MAX_PASSWORD_LENGTH, len(chars)))

x = np.zeros((len(inputs), MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), MAX_PASSWORD_LENGTH, len(chars)), dtype=np.bool)

sleep(1)
for i, element in enumerate(tqdm(inputs, desc='inputs')):
    x[i] = c_table.encode(element, MAX_PASSWORD_LENGTH)
for i, element in enumerate(tqdm(targets, desc='targets')):
    y[i] = c_table.encode(element, MAX_PASSWORD_LENGTH)

indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

np.savez_compressed('/tmp/x_y.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

print('Done... File is /tmp/x_y.npz')

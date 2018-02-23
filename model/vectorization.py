import numpy as np
from tqdm import tqdm

from model.data_gen import LazyDataLoader, build_vocabulary, get_token_indices, get_chars_and_ctable

build_vocabulary()

print('Vectorization...')

DATA_LOADER = LazyDataLoader()

INPUT_MAX_LEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

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

x = np.zeros((len(inputs), INPUT_MAX_LEN, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), OUTPUT_MAX_LEN, len(chars)), dtype=np.bool)

for i, element in enumerate(tqdm(inputs, desc='inputs')):
    x[i] = c_table.encode(element, INPUT_MAX_LEN)
for i, element in enumerate(tqdm(targets, desc='targets')):
    y[i] = c_table.encode(element, OUTPUT_MAX_LEN)

indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

np.savez_compressed('x_y.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

print('Done... File is x_y.npz')

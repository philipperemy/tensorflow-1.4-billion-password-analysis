import numpy as np
from tqdm import tqdm

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
for i in tqdm(range(TRAINING_SIZE), desc='Generating inputs and targets'):
    x_, y_ = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    inputs.append(x_)
    targets.append(y_)

np.savez_compressed('/tmp/x_y.npz', inputs=inputs, targets=targets)

print('Done... File is /tmp/x_y.npz')

import argparse

import numpy as np
import os
from tqdm import tqdm

from data_gen import LazyDataLoader, build_vocabulary, get_chars_and_ctable

parser = argparse.ArgumentParser('Data Encoding Tool.')
parser.add_argument('--training_filename', type=os.path.expanduser,
                    help='Result of run_data_processing.py. '
                         'Something like: /home/premy/BreachCompilationAnalysis/edit-distances/1.csv',
                    required=True)
# parser.add_argument('--encoding_output_folder', type=str, help='Will be used for training')

arg_p = parser.parse_args()

print('Building vocabulary...')

build_vocabulary(arg_p.training_filename)

print('Vectorization...')

DATA_LOADER = LazyDataLoader(arg_p.training_filename)

_, _, training_records_count = DATA_LOADER.statistics()

# TOKEN_INDICES = get_token_indices()

chars, c_table = get_chars_and_ctable()

inputs = []
targets = []
print('Generating data...')
for i in tqdm(range(training_records_count), desc='Generating inputs and targets'):
    x_, y_ = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    inputs.append(x_)
    targets.append(y_)

np.savez_compressed('/tmp/x_y.npz', inputs=inputs, targets=targets)

print('Done... File is /tmp/x_y.npz')

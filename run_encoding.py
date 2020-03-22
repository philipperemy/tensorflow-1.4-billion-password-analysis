import argparse

import numpy as np
import os
from tqdm import tqdm

from batcher import LazyDataLoader, build_vocabulary, Batcher

parser = argparse.ArgumentParser('Data Encoding Tool.')
parser.add_argument('--training_filename', type=os.path.expanduser,
                    help='Result of run_data_processing.py. '
                         'Something like: /home/premy/BreachCompilationAnalysis/edit-distances/1.csv',
                    required=True)
# parser.add_argument('--encoding_output_folder', type=str, help='Will be used for training')

args = parser.parse_args()

print('Building vocabulary...')

build_vocabulary(args.training_filename)

print('Vectorization...')

DATA_LOADER = LazyDataLoader(args.training_filename)
SEQ = Batcher()

_, _, training_records_count = DATA_LOADER.statistics()

# token_indices = get_token_indices()

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

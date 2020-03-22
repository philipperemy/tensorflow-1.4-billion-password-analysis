import pickle
from collections import Counter

import numpy as np
import os
from tqdm import tqdm

from utils import TMP_DIR


class Batcher:
    # Maximum password length. Passwords greater than this length will be discarded during the encoding phase.
    ENCODING_MAX_PASSWORD_LENGTH = 12

    # Maximum number of characters for encoding. By default, we use the 80 most frequent characters and
    # we bin the other ones in a OOV (out of vocabulary) group.
    ENCODING_MAX_SIZE_VOCAB = 80

    @staticmethod
    def build(training_filename):
        print('Building vocabulary...')
        build_vocabulary(training_filename)
        print('Vectorization...')
        data_loader = LazyDataLoader(training_filename)
        _, _, training_records_count = data_loader.statistics()
        inputs = []
        targets = []
        print('Generating data...')
        for _ in tqdm(range(training_records_count), desc='Generating inputs and targets'):
            x_, y_ = data_loader.next()
            # Pad the data with spaces such that it is always MAXLEN.
            inputs.append(x_)
            targets.append(y_)

        np.savez_compressed('/tmp/x_y.npz', inputs=inputs, targets=targets)

        print('Done... File is /tmp/x_y.npz')

    @staticmethod
    def load():
        if not os.path.exists('/tmp/x_y.npz'):
            raise Exception('Please run the vectorization script before.')

        print('Loading data from prefetch...')
        data = np.load('/tmp/x_y.npz')
        inputs = data['inputs']
        targets = data['targets']

        print('Data:')
        print(inputs.shape)
        print(targets.shape)
        return inputs, targets

    def __init__(self):
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        self.token_indices = os.path.join(TMP_DIR, 'token_indices.pkl')
        self.indices_token = os.path.join(TMP_DIR, 'indices_token.pkl')

        try:
            self.chars, self.c_table = self.get_chars_and_ctable()
        except FileNotFoundError:
            raise Exception('Run first run_encoding.py to generate the required files.')

    def chars_len(self):
        return len(self.chars)

    def get_indices_token(self):
        return pickle.load(open(self.indices_token, 'rb'))

    def get_token_indices(self):
        return pickle.load(open(self.token_indices, 'rb'))

    def get_vocab_size(self):
        return len(self.get_token_indices())

    def get_chars_and_ctable(self):
        chars = ''.join(list(self.get_token_indices().values()))
        ctable = CharacterTable(chars)
        return chars, ctable

    def write(self, vocabulary_sorted_list):
        token_indices = dict((c, i) for (c, i) in enumerate(vocabulary_sorted_list))
        indices_token = dict((i, c) for (c, i) in enumerate(vocabulary_sorted_list))
        assert len(token_indices) == len(indices_token)

        with open(self.token_indices, 'wb') as w:
            pickle.dump(obj=token_indices, file=w)

        with open(self.indices_token, 'wb') as w:
            pickle.dump(obj=indices_token, file=w)

        print(f'Done... File is {self.token_indices}.')
        print(f'Done... File is {self.indices_token}.')

    def decode(self, char, calc_argmax=True):
        return self.c_table.decode(char, calc_argmax)

    def encode(self, elt, num_rows=ENCODING_MAX_PASSWORD_LENGTH):
        self.c_table.encode(elt, num_rows)


def discard_password(password):
    return len(password) > Batcher.ENCODING_MAX_PASSWORD_LENGTH or ' ' in password


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i in range(num_rows):
            try:
                c = C[i]
                if c not in self.char_indices:
                    x[i, self.char_indices['？']] = 1
                else:
                    x[i, self.char_indices[c]] = 1
            except IndexError:
                x[i, self.char_indices[' ']] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def build_vocabulary(training_filename):
    sed = Batcher()
    vocabulary = Counter()
    print('Reading file {}.'.format(training_filename))
    with open(training_filename, 'r', encoding='utf8', errors='ignore') as r:
        for s in tqdm(r.readlines(), desc='Build Vocabulary'):
            _, x, y = s.strip().split(' ||| ')
            if discard_password(y) or discard_password(x):
                continue
            vocabulary += Counter(list(y + x))
    vocabulary_sorted_list = sorted(dict(vocabulary.most_common(ENCODING_MAX_SIZE_VOCAB)).keys())
    oov_char = '？'
    pad_char = ' '
    print('Out of vocabulary (OOV) char is {}.'.format(oov_char))
    print('Pad char is "{}".'.format(pad_char))
    vocabulary_sorted_list.append(oov_char)  # out of vocabulary.
    vocabulary_sorted_list.append(pad_char)  # pad char.
    print('Vocabulary = ' + ' '.join(vocabulary_sorted_list))
    sed.write(vocabulary_sorted_list)


def stream_from_file(training_filename):
    with open(training_filename, 'rb') as r:
        for l in r.readlines():
            _, x, y = l.decode('utf8').strip().split(' ||| ')
            if discard_password(y) or discard_password(x):
                continue
            yield x.strip(), y.strip()


class LazyDataLoader:
    def __init__(self, training_filename):
        self.training_filename = training_filename
        self.stream = stream_from_file(self.training_filename)

    def next(self):
        try:
            return next(self.stream)
        except:
            self.stream = stream_from_file(self.training_filename)
            return self.next()

    def statistics(self):
        max_len_value_x = 0
        max_len_value_y = 0
        num_lines = 0
        self.stream = stream_from_file(self.training_filename)
        for x, y in self.stream:
            max_len_value_x = max(max_len_value_x, len(x))
            max_len_value_y = max(max_len_value_y, len(y))
            num_lines += 1

        print('max_len_value_x =', max_len_value_x)
        print('max_len_value_y =', max_len_value_y)
        print('num_lines =', num_lines)
        return max_len_value_x, max_len_value_y, num_lines
import json
import os
import tempfile
from collections import Counter

import numpy as np
from tqdm import tqdm

from utils import stream_from_file


class Batcher:
    TMP_DIR = tempfile.gettempdir()

    SEP = '\t'

    # Maximum password length. Passwords greater than this length will be discarded during the encoding phase.
    ENCODING_MAX_PASSWORD_LENGTH = 12

    # Maximum number of characters for encoding. By default, we use the 80 most frequent characters and
    # we bin the other ones in a OOV (out of vocabulary) group.
    ENCODING_MAX_SIZE_VOCAB = 80

    INPUTS_TARGETS_FILENAME = os.path.join(TMP_DIR, 'model_inputs.npz')

    OOV_CHAR = '？'
    PAD_CHAR = ' '

    def __init__(self):
        if not os.path.exists(self.TMP_DIR):
            os.makedirs(self.TMP_DIR)

        self.token_indices = os.path.join(self.TMP_DIR, 'token_indices.json')
        self.indices_token = os.path.join(self.TMP_DIR, 'indices_token.json')

        if os.path.exists(Batcher.INPUTS_TARGETS_FILENAME):
            print('Loading data from prefetch...')
            data = np.load(Batcher.INPUTS_TARGETS_FILENAME)
            inputs = data['inputs']
            targets = data['targets']
            print('Data:')
            print(inputs.shape)
            print(targets.shape)
            self.inputs, self.targets = inputs, targets
        else:
            self.inputs, self.targets = None, None
        if os.path.exists(self.token_indices):
            print('Loading the character table...')
            self.chars, self.c_table = self.get_chars_and_ctable()
        else:
            self.chars, self.c_table = None, None

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
        # TODO: this batcher is not efficient at all.
        for _ in tqdm(range(training_records_count), desc='Generating inputs and targets'):
            x_, y_ = data_loader.next()
            # Pad the data with spaces such that it is always MAXLEN.
            inputs.append(x_)
            targets.append(y_)

        np.savez_compressed(Batcher.INPUTS_TARGETS_FILENAME, inputs=inputs, targets=targets)
        print(f'Done... File is {Batcher.INPUTS_TARGETS_FILENAME}.')

    def chars_len(self):
        return len(self.chars)

    def get_indices_token(self):
        with open(self.indices_token, 'r') as r:
            return json.load(r)

    def get_token_indices(self):
        with open(self.token_indices, 'r') as r:
            return json.load(r)

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

        with open(self.token_indices, 'w') as w:
            json.dump(obj=token_indices, fp=w, indent=2)

        with open(self.indices_token, 'w') as w:
            json.dump(obj=indices_token, fp=w, indent=2)

        print(f'Done... File is {self.token_indices}.')
        print(f'Done... File is {self.indices_token}.')

    def decode(self, char, calc_argmax=True):
        return self.c_table.decode(char, calc_argmax)

    def encode(self, elt, num_rows=ENCODING_MAX_PASSWORD_LENGTH):
        return self.c_table.encode(elt, num_rows)


def discard_password(password):
    return len(password) > Batcher.ENCODING_MAX_PASSWORD_LENGTH or ' ' in password


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """
        Initialize character table.
        Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, s, num_rows):
        """
        One hot encode given string s.
        Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i in range(num_rows):
            try:
                c = s[i]
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
            try:
                _, x, y = s.strip().split(Batcher.SEP)
                if discard_password(y) or discard_password(x):
                    continue
                vocabulary += Counter(list(y + x))
            except Exception:
                print('Error during encoding.')
    vocabulary_sorted_list = sorted(dict(vocabulary.most_common(sed.ENCODING_MAX_SIZE_VOCAB)).keys())

    print('Out of vocabulary (OOV) char is {}.'.format(sed.OOV_CHAR))
    print('Pad char is "{}".'.format(sed.PAD_CHAR))
    vocabulary_sorted_list.append(sed.OOV_CHAR)  # out of vocabulary.
    vocabulary_sorted_list.append(sed.PAD_CHAR)  # pad char.
    print('Vocabulary = ' + ' '.join(vocabulary_sorted_list))
    sed.write(vocabulary_sorted_list)


class LazyDataLoader:

    def __init__(self, edit_distance_file):
        self.edit_distance_file = edit_distance_file
        self.stream = self.init_stream()

    def init_stream(self):
        return stream_from_file(self.edit_distance_file, sep=Batcher.SEP, discard_fun=discard_password)

    def next(self):
        try:
            return next(self.stream)
        except Exception:
            self.stream = self.init_stream()
            return self.next()

    def statistics(self):
        max_len_value_x = 0
        max_len_value_y = 0
        num_lines = 0
        self.stream = self.init_stream()
        for x, y in self.stream:
            max_len_value_x = max(max_len_value_x, len(x))
            max_len_value_y = max(max_len_value_y, len(y))
            num_lines += 1

        print('max_len_value_x =', max_len_value_x)
        print('max_len_value_y =', max_len_value_y)
        print('num_lines =', num_lines)
        return max_len_value_x, max_len_value_y, num_lines

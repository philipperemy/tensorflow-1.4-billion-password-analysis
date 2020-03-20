import pickle
from collections import Counter

import numpy as np
import os
from tqdm import tqdm

from train_constants import ENCODING_MAX_PASSWORD_LENGTH, ENCODING_MAX_SIZE_VOCAB

TMP_DIR = 'tmp'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

TOKEN_INDICES = os.path.join(TMP_DIR, 'token_indices.pkl')
INDICES_TOKEN = os.path.join(TMP_DIR, 'indices_token.pkl')


def get_indices_token():
    return pickle.load(open(INDICES_TOKEN, 'rb'))


def get_token_indices():
    return pickle.load(open(TOKEN_INDICES, 'rb'))


def get_vocab_size():
    return len(get_token_indices())


def discard_password(password):
    return len(password) > ENCODING_MAX_PASSWORD_LENGTH or ' ' in password


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


def get_chars_and_ctable():
    chars = ''.join(list(get_token_indices().values()))
    ctable = CharacterTable(chars)
    return chars, ctable


def build_vocabulary(training_filename):
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
    token_indices = dict((c, i) for (c, i) in enumerate(vocabulary_sorted_list))
    indices_token = dict((i, c) for (c, i) in enumerate(vocabulary_sorted_list))
    assert len(token_indices) == len(indices_token)

    with open(TOKEN_INDICES, 'wb') as w:
        pickle.dump(obj=token_indices, file=w)

    with open(INDICES_TOKEN, 'wb') as w:
        pickle.dump(obj=indices_token, file=w)

    print(f'Done... File is {TOKEN_INDICES}.')
    print(f'Done... File is {INDICES_TOKEN}.')


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


if __name__ == '__main__':
    # how to use it.
    ldl = LazyDataLoader('/home/premy/BreachCompilationAnalysis/edit-distances/1.csv')
    print(ldl.statistics())
    while True:
        print(ldl.next())

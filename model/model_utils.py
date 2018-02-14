import pickle

import numpy as np

from constants import EDIT_DISTANCE_FILENAME


def get_INDICES_TOKEN():
    return pickle.load(open('indices_token.pkl', 'rb'))


def get_TOKEN_INDICES():
    return pickle.load(open('token_indices.pkl', 'rb'))


def get_VOCAB_SIZE():
    return len(get_TOKEN_INDICES())


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
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
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
    chars = ''.join(list(get_TOKEN_INDICES().values()))
    ctable = CharacterTable(chars)
    return chars, ctable


def build_vocabulary():
    vocabulary = set()
    with open(EDIT_DISTANCE_FILENAME, 'rb') as r:
        for l in r.readlines():
            line_id, y, x = l.decode('utf8').strip().split(' ||| ')
            for element in list(y):
                vocabulary.add(element)
            for element in list(x):
                vocabulary.add(element)
    vocabulary = sorted(list(vocabulary))
    print(vocabulary)
    token_indices = dict((c, i) for (c, i) in enumerate(vocabulary))
    indices_token = dict((i, c) for (c, i) in enumerate(vocabulary))

    with open('token_indices.pkl', 'wb') as w:
        pickle.dump(obj=token_indices, file=w)

    with open('indices_token.pkl', 'wb') as w:
        pickle.dump(obj=indices_token, file=w)

    print('Done... File is token_indices.pkl')
    print('Done... File is indices_token.pkl')


if __name__ == '__main__':
    build_vocabulary()

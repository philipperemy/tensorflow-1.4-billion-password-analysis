# -*- coding: utf-8 -*-
import argparse
from collections import Counter

import Levenshtein
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from batcher import Batcher


def get_script_arguments():
    parser = argparse.ArgumentParser(description='Training a password model.')
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    return args


def gen_large_chunk_single_thread(sed: Batcher, inputs_, targets_, chunk_size):
    # make it simple now.
    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    x = np.zeros((chunk_size, sed.ENCODING_MAX_PASSWORD_LENGTH, sed.chars_len()), dtype=float)
    y2 = []
    y1 = []

    for i in range(len(sub_inputs)):
        # ed = 1
        edit_dist = Levenshtein.editops(sub_inputs[i], sub_targets[i])[0]
        if edit_dist[0] == 'insert':
            tt = [1, 0, 0]
            assert edit_dist[1] == edit_dist[2]
            char_changed = sub_targets[i][edit_dist[1]]
        elif edit_dist[0] == 'replace':
            tt = [0, 1, 0]
            assert edit_dist[1] == edit_dist[2]
            char_changed = sub_targets[i][edit_dist[1]]
        elif edit_dist[0] == 'delete':
            tt = [0, 0, 1]
            assert edit_dist[1] == edit_dist[2]
            char_changed = sub_inputs[i][edit_dist[1]]
        else:
            raise Exception('Impossible.')
        y1.append(tt)
        y2.append(sed.encode(char_changed, len(char_changed)).squeeze())

    y1 = np.array(y1)
    y2 = np.array(y2)

    for i_, element in enumerate(sub_inputs):
        x[i_] = sed.encode(element)

    split_at = int(len(x) * 0.9)
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train_1, y_val_1) = y1[:split_at], y1[split_at:]
    (y_train_2, y_val_2) = y2[:split_at], y2[split_at:]
    val_sub_targets = sub_targets[split_at:]
    val_sub_inputs = sub_inputs[split_at:]

    return x_train, y_train_1, y_train_2, x_val, y_val_1, y_val_2, val_sub_inputs, val_sub_targets


def predict_top_most_likely_passwords_monte_carlo(sed: Batcher, model_, rowx_, n_, mc_samples=10000):
    samples = predict_top_most_likely_passwords(sed, model_, rowx_, mc_samples)
    return dict(Counter(samples).most_common(n_)).keys()


def predict_top_most_likely_passwords(sed: Batcher, model_, rowx_, n_):
    p_ = model_.predict(rowx_, batch_size=32, verbose=0)[0]
    most_likely_passwords = []
    for ii in range(n_):
        # of course should take the edit distance constraint.
        pa = np.array([np.random.choice(a=range(sed.ENCODING_MAX_SIZE_VOCAB + 2), size=1, p=p_[jj, :])
                       for jj in range(sed.ENCODING_MAX_PASSWORD_LENGTH)]).flatten()
        most_likely_passwords.append(sed.decode(pa, calc_argmax=False))
    return most_likely_passwords
    # Could sample 1000 and take the most_common()


def get_model(hidden_size, num_chars):
    i = Input(shape=(Batcher.ENCODING_MAX_PASSWORD_LENGTH, num_chars))
    x = LSTM(hidden_size)(i)
    x = Dense(Batcher.ENCODING_MAX_PASSWORD_LENGTH * num_chars, activation='relu')(x)

    # ADD, DEL, SUB
    o1 = Dense(3, activation='softmax', name='op')(x)
    o2 = Dense(num_chars, activation='softmax', name='char')(x)

    return Model(inputs=[i], outputs=[o1, o2])


def main():
    inputs, targets = Batcher.load()
    print('Data:')
    print(inputs.shape)
    print(targets.shape)

    args = get_script_arguments()

    # Try replacing GRU.
    batcher = Batcher()
    model = get_model(args.hidden_size, batcher.chars_len())

    losses = {
        "op": "categorical_crossentropy",
        "char": "categorical_crossentropy",
    }

    model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
    model.summary()

    for iteration in range(1, int(1e9)):
        ppp = gen_large_chunk_single_thread(batcher, inputs, targets, chunk_size=args.batch_size * 500)
        x_train, y_train_1, y_train_2, x_val, y_val_1, y_val_2, val_sub_inputs, val_sub_targets = ppp
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x=x_train, y=[y_train_1, y_train_2], batch_size=args.batch_size, epochs=5,
                  validation_data=(x_val, [y_val_1, y_val_2]))
        rowx, correct, previous = x_val, val_sub_targets, val_sub_inputs  # replace by x_val, y_val
        op, char = model.predict(rowx, verbose=0)
        q = list(batcher.decode(char))
        op = op.argmax(axis=1)
        decoded_op = []
        for opp in op:
            if opp == 0:
                decoded_op.append('insert')
            elif opp == 1:
                decoded_op.append('replace')
            else:
                decoded_op.append('delete')
        # guess = c_table.decode(preds[0], calc_argmax=False)
        # top_passwords = predict_top_most_likely_passwords_monte_carlo(model, rowx, 100)
        # p = model.predict(rowx, batch_size=32, verbose=0)[0]
        # p.shape (12, 82)
        # [np.random.choice(a=range(82), size=1, p=p[i, :]) for i in range(12)]
        # s = [np.random.choice(a=range(82), size=1, p=p[i, :])[0] for i in range(12)]
        # c_table.decode(s, calc_argmax=False)
        # Could sample 1000 and take the most_common()
        for c, p, qq, dd in zip(correct[0:10], previous[0:10], q[0:10], decoded_op[0:10]):
            print('y      :', c)
            print('x      :', p)
            print('predict char :', qq)
            print('predict op   :', dd)
            print('---------------------')

            # if correct == guess:
            # if correct.strip() in [vv.strip() for vv in top_passwords]:
            #     print(colors.ok + '☑' + colors.close)
            # else:
            #     print(colors.fail + '☒' + colors.close)
            # print('top    :', ', '.join(top_passwords))
            # print('---')


if __name__ == '__main__':
    main()

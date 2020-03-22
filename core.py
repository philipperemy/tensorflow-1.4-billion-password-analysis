import itertools
import json
import os
from collections import Counter
from glob import glob
from itertools import combinations

import Levenshtein
import editdistance
import numpy as np
from slugify import slugify
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tqdm import tqdm

from batcher import Batcher
from utils import ensure_dir, ensure_dir_for_file, create_new_dir


class Callback:
    def __init__(self):
        pass

    def call(self, emails_passwords):
        # emails_passwords = list of tuples
        pass


class ReducePasswordsOnSimilarEmailsCallback(Callback):
    NAME = 'reduce-passwords-on-similar-emails'

    def __init__(self, persisted_filename, output_folder):
        super().__init__()
        self.cache = {}
        self.cache_key_edit_distance_keep_user_struct = {}
        self.cache_key_edit_distance_list = {}
        self.filename = persisted_filename
        self.output_folder = output_folder

    def _finalize_cache(self):
        keys = list(self.cache.keys())
        for key in keys:
            orig_password_list = list(self.cache[key])
            del self.cache[key]
            if len(orig_password_list) > 1:
                shp = list(find_shortest_hamiltonian_path_in_complete_graph(orig_password_list, False))
                if len(shp) == 0:
                    continue  # shortest_hamiltonian_path did not return well.

                edit_distances = []
                for a, b in zip(shp, shp[1:]):
                    ed = editdistance.eval(a, b)
                    edit_distances.append(ed)
                    if ed not in self.cache_key_edit_distance_list:
                        self.cache_key_edit_distance_list[ed] = []
                    self.cache_key_edit_distance_list[ed].append((a, b))

                self.cache[key] = {}
                self.cache[key]['password'] = shp
                self.cache[key]['edit_distance'] = [0] + edit_distances
                mean_edit_distance_key = float('{0:.2f}'.format(np.mean(edit_distances)))
                if mean_edit_distance_key not in self.cache_key_edit_distance_keep_user_struct:
                    self.cache_key_edit_distance_keep_user_struct[mean_edit_distance_key] = []
                new_elt = {'password': self.cache[key]['password'],
                           'edit_distance': self.cache[key]['edit_distance'],
                           'email': key}
                self.cache_key_edit_distance_keep_user_struct[mean_edit_distance_key].append(new_elt)

    def call(self, emails_passwords):
        for (email, password) in emails_passwords:
            if email not in self.cache:
                self.cache[email] = set()
            self.cache[email].add(password.strip())

    def persist(self):
        self._finalize_cache()
        output_file = self.filename + '_per_user.json'
        ensure_dir_for_file(output_file)
        with open(output_file, 'w') as w:
            json.dump(fp=w, obj=self.cache_key_edit_distance_keep_user_struct, indent=4, sort_keys=True,
                      ensure_ascii=False)

        for edit_distance in sorted(self.cache_key_edit_distance_list):
            def csv_line_format(x):
                return str(edit_distance) + Batcher.SEP + x[0] + Batcher.SEP + x[1] + '\n'

            output_dir = os.path.join(os.path.expanduser(self.output_folder), 'edit-distances')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            csv_file = os.path.join(output_dir, str(edit_distance) + '.csv')
            # print('Updating: ' + csv_file)
            with open(csv_file, encoding='utf8', mode='a') as w:
                password_pairs = self.cache_key_edit_distance_list[edit_distance]
                lines = list(map(csv_line_format, password_pairs))
                w.writelines(lines)


def find_shortest_hamiltonian_path_in_complete_graph(passwords, debug=True):
    # passwords = ['hello1', 'hello22', 'h@llo22', 'h@llo223']
    #     print(find_shortest_hamiltonian_path_in_complete_graph(passwords, False))
    # complexity is paramount! This script runs in factorial(n)

    if len(passwords) > 6:  # 6! = 720 combinations.
        return []

    map_edit_distance = {}

    # shortest hamiltonian path in complete graph. NP-complete.
    for combo in combinations(passwords, 2):  # 2 for pairs, 3 for triplets, etc
        ed = editdistance.eval(combo[0], combo[1])
        if debug:
            print(combo[0], combo[1], ed)
        map_edit_distance[(combo[0], combo[1])] = ed
        map_edit_distance[(combo[1], combo[0])] = ed

    # factorial(n)
    # permutations = list(itertools.permutations(passwords))
    permutations = list(filter(lambda x: len(x[0]) == min([len(a) for a in x]),
                               list(itertools.permutations(passwords))))

    all_solutions = {}
    for permutation in permutations:
        full_ed = 0
        for a, b in zip(permutation, permutation[1:]):
            full_ed += map_edit_distance[(a, b)]

        if debug:
            print(full_ed, permutation)

        if full_ed not in all_solutions:
            all_solutions[full_ed] = []
        all_solutions[full_ed].append(permutation)

    if debug:
        print(json.dumps(all_solutions, indent=2))

    lowest_ed = sorted(all_solutions.keys())[0]

    if debug:
        print(lowest_ed)
    # we consider that the first password is the easiest one (at least the shortest one).
    best_solutions = all_solutions[lowest_ed]

    if debug:
        print(best_solutions)

    final_solution = best_solutions[np.argmin([len(bs[0]) for bs in best_solutions])]

    if debug:
        print(final_solution)

    return final_solution


def preprocess(breach_compilation_folder, output_folder, max_num_files):
    on_file_read_call_back_class = ReducePasswordsOnSimilarEmailsCallback
    all_filenames = glob(os.path.expanduser(breach_compilation_folder) + '/**/*', recursive=True)
    all_filenames = sorted(list(filter(os.path.isfile, all_filenames)))
    callback_class_name = on_file_read_call_back_class.NAME
    callback_output_dir = os.path.join(output_folder, callback_class_name)
    create_new_dir(output_folder)

    print('FOUND: {0} unique files in {1}.'.format(len(all_filenames), breach_compilation_folder))
    if max_num_files is not None:
        print('TRUNCATE DATASET TO: {0} files.'.format(max_num_files))
        all_filenames = all_filenames[0:max_num_files]

    bar = tqdm(all_filenames)
    for current_filename in bar:
        if os.path.isfile(current_filename):
            suffix = slugify(current_filename.split('data')[-1])
            output_filename = os.path.join(callback_output_dir, suffix)
            callback = on_file_read_call_back_class(output_filename, output_folder)
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
                emails_passwords = extract_emails_and_passwords(lines)
                callback.call(emails_passwords)
            bar.set_description('Processing {0:,} passwords for {1}'.format(len(callback.cache), current_filename))
            callback.persist()
    bar.close()
    print('DONE. SUCCESS.')
    print('OUTPUT: Dataset was generated at: {0}.'.format(output_folder))


def build_encodings(training_filename):
    Batcher.build(training_filename)


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


def train(hidden_size, batch_size):
    # Try replacing GRU.
    batcher = Batcher()
    inputs, targets = Batcher.load()
    print('Data:')
    print(inputs.shape)
    print(targets.shape)

    model = get_model(hidden_size, batcher.chars_len())

    losses = {
        "op": "categorical_crossentropy",
        "char": "categorical_crossentropy",
    }

    model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
    model.summary()

    for iteration in range(1, int(1e9)):
        ppp = gen_large_chunk_single_thread(batcher, inputs, targets, chunk_size=batch_size * 500)
        x_train, y_train_1, y_train_2, x_val, y_val_1, y_val_2, val_sub_inputs, val_sub_targets = ppp
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x=x_train, y=[y_train_1, y_train_2], batch_size=batch_size, epochs=5,
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


def extract_emails_and_passwords(txt_lines):
    emails_passwords = []
    for txt_line in txt_lines:
        try:
            if '@' in txt_line:  # does it contain an email address?
                if all([char in txt_line for char in [':', ';']]):  # which separator is it? : or ;?
                    separator = ':'
                elif ':' in txt_line:  # '_---madc0w---_@live.com:iskandar89
                    separator = ':'
                elif ';' in txt_line:  # '_---lelya---_@mail.ru;ol1391ga
                    separator = ';'
                else:
                    continue

                strip_txt_line = txt_line.strip()
                email, password = strip_txt_line.split(separator)
                emails_passwords.append((email, password))
        except Exception:
            pass
    return emails_passwords

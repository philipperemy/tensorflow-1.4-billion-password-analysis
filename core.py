import itertools
import json
import os
from collections import Counter, defaultdict
from glob import glob
from itertools import combinations

import Levenshtein
import editdistance
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tqdm import tqdm

from batcher import Batcher
from utils import create_new_dir, ensure_dir


class EditDistanceParser:

    def __init__(self, output_dir):
        self.cache = defaultdict(set)
        self.ed_to_password_list_map = defaultdict(list)
        self.output_dir = output_dir

    def _finalize_cache(self):
        for key, password_list in self.cache.items():
            if len(password_list) > 1:
                shp = find_shortest_hamiltonian_path_in_complete_graph(password_list, debug=False)
                if len(shp) == 0:
                    continue  # shortest_hamiltonian_path did not return well.
                edit_distances = []
                for a, b in zip(shp, shp[1:]):
                    ed = editdistance.eval(a, b)
                    edit_distances.append(ed)
                    self.ed_to_password_list_map[ed].append((a, b))

    def call(self, emails_passwords):
        for (email, password) in emails_passwords:
            self.cache[email].add(password.strip())

    def flush(self):
        self.cache = defaultdict(set)
        self.ed_to_password_list_map = defaultdict(list)

    def persist(self):
        self._finalize_cache()
        for edit_distance in sorted(self.ed_to_password_list_map):
            output_dir = os.path.join(os.path.expanduser(self.output_dir), 'edit-distances')
            ensure_dir(output_dir)
            csv_file = os.path.join(output_dir, str(edit_distance) + '.csv')
            with open(csv_file, encoding='utf8', mode='a') as w:
                password_pairs = self.ed_to_password_list_map[edit_distance]
                lines = [str(edit_distance) + Batcher.SEP + x[0] + Batcher.SEP + x[1] + '\n' for x in password_pairs]
                w.writelines(lines)
        self.flush()


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

    final_solution = best_solutions[int(np.argmin([len(bs[0]) for bs in best_solutions]))]

    if debug:
        print(final_solution)

    return list(final_solution)


def preprocess(breach_compilation_folder, output_dir, max_num_files):
    bc_dir = os.path.expanduser(breach_compilation_folder)
    all_filenames = glob(bc_dir + '/**/*', recursive=True)
    all_filenames = [f for f in list(filter(os.path.isfile, all_filenames)) if os.path.isfile(f)]
    create_new_dir(output_dir)
    print(f'Found {len(all_filenames)} files in {bc_dir}.')
    if max_num_files is not None:
        all_filenames = all_filenames[:max_num_files]
    edp = EditDistanceParser(output_dir)
    with tqdm(all_filenames) as bar:
        for current_filename in bar:
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
            emails_passwords = extract_emails_and_passwords(lines)
            edp.call(emails_passwords)
            edp.persist()
    print('DONE. SUCCESS.')
    print(f'OUTPUT: Dataset was generated at: {output_dir}.')


def build_encodings(training_filename):
    Batcher.build(training_filename)


def gen_large_chunk_single_thread(sed: Batcher, inputs_, targets_, chunk_size):
    # make it simple now.
    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    n = len(sub_inputs)
    x = np.zeros((chunk_size, sed.ENCODING_MAX_PASSWORD_LENGTH, sed.chars_len()), dtype=float)
    y2_char = np.zeros(shape=(n, sed.chars_len()))
    y1_op = np.zeros(shape=(n, 3))

    for i in range(n):
        # ed = 1
        edit_dist = Levenshtein.editops(sub_inputs[i], sub_targets[i])[0]
        op = edit_dist[0]
        assert edit_dist[1] == edit_dist[2]
        if op == 'insert':
            op_encoding = [1, 0, 0]
            char_changed = sub_targets[i][edit_dist[1]]
        elif op == 'replace':
            op_encoding = [0, 1, 0]
            char_changed = sub_targets[i][edit_dist[1]]
        elif op == 'delete':
            op_encoding = [0, 0, 1]
            char_changed = sub_inputs[i][edit_dist[1]]
        else:
            raise Exception('Unsupported op.')
        y1_op[i] = op_encoding
        y2_char[i] = sed.encode(char_changed, 1)[0]

    for i, element in enumerate(sub_inputs):
        x[i] = sed.encode(element)

    split_at = int(len(x) * 0.9)
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train_1, y_val_1) = y1_op[:split_at], y1_op[split_at:]
    (y_train_2, y_val_2) = y2_char[:split_at], y2_char[split_at:]
    val_sub_targets = sub_targets[split_at:]
    val_sub_inputs = sub_inputs[split_at:]

    return x_train, y_train_1, y_train_2, x_val, y_val_1, y_val_2, val_sub_inputs, val_sub_targets


def predict_top_most_likely_passwords_monte_carlo(sed: Batcher, model, row_x, n, mc_samples=10000):
    samples = predict_top_most_likely_passwords(sed, model, row_x, mc_samples)
    return dict(Counter(samples).most_common(n)).keys()


def predict_top_most_likely_passwords(sed: Batcher, model, row_x, n):
    p = model.predict(row_x, batch_size=32, verbose=0)[0]
    most_likely_passwords = []
    for i in range(n):
        # of course should take the edit distance constraint.
        pa = np.array([np.random.choice(a=range(sed.ENCODING_MAX_SIZE_VOCAB + 2), size=1, p=p[j, :])
                       for j in range(sed.ENCODING_MAX_PASSWORD_LENGTH)]).flatten()
        most_likely_passwords.append(sed.decode(pa, calc_argmax=False))
    return most_likely_passwords


def get_model(hidden_size, num_chars):
    i = Input(shape=(Batcher.ENCODING_MAX_PASSWORD_LENGTH, num_chars))
    x = LSTM(hidden_size)(i)
    x = Dense(Batcher.ENCODING_MAX_PASSWORD_LENGTH * num_chars, activation='relu')(x)
    # ADD, DEL, SUB
    o1 = Dense(3, activation='softmax', name='op')(x)
    o2 = Dense(num_chars, activation='softmax', name='char')(x)
    return Model(inputs=[i], outputs=[o1, o2])


def train(hidden_size, batch_size):
    batcher = Batcher()
    print('Data:')
    print(batcher.inputs.shape)
    print(batcher.targets.shape)

    model = get_model(hidden_size, batcher.chars_len())

    model.compile(loss={'op': 'categorical_crossentropy', 'char': 'categorical_crossentropy'},
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    while True:
        ppp = gen_large_chunk_single_thread(batcher, batcher.inputs, batcher.targets, chunk_size=batch_size * 500)
        x_train, y_train_1, y_train_2, x_val, y_val_1, y_val_2, val_sub_inputs, val_sub_targets = ppp
        print()
        model.fit(x=x_train, y=[y_train_1, y_train_2], batch_size=batch_size, epochs=20,
                  validation_data=(x_val, [y_val_1, y_val_2]))
        row_x, password_target, password_input = x_val, val_sub_targets, val_sub_inputs
        ops, char = model.predict(row_x, verbose=0)
        predicted_chars = list(batcher.decode(char))
        ops = ops.argmax(axis=1)
        decoded_op = []
        for op in ops:
            if op == 0:
                decoded_op.append('insert')
            elif op == 1:
                decoded_op.append('replace')
            else:
                decoded_op.append('delete')
        # guess = c_table.decode(preds[0], calc_argmax=False)
        # top_passwords = predict_top_most_likely_passwords_monte_carlo(model, row_x, 100)
        # p = model.predict(row_x, batch_size=32, verbose=0)[0]
        # p.shape (12, 82)
        # [np.random.choice(a=range(82), size=1, p=p[i, :]) for i in range(12)]
        # s = [np.random.choice(a=range(82), size=1, p=p[i, :])[0] for i in range(12)]
        # c_table.decode(s, calc_argmax=False)
        # Could sample 1000 and take the most_common()
        for i, (x, y, pc, po) in enumerate(zip(password_input, password_target, predicted_chars, decoded_op)):
            print('x            :', x)
            print('y            :', y)
            print('predict char :', pc)
            print('predict op   :', po)
            print('---------------------')
            if i >= 10:
                break

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
                if ':' in txt_line:  # '_---madc0w---_@live.com:iskandar89
                    separator = ':'
                elif ';' in txt_line:  # '_---lelya---_@mail.ru;ol1391ga
                    separator = ';'
                else:
                    continue
                strip_txt_line = txt_line.strip()
                email, password = strip_txt_line.split(separator, 1)
                emails_passwords.append((email, password))
        except Exception:
            pass
    return emails_passwords

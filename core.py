import itertools
import json
from glob import glob
from itertools import combinations

import editdistance
import numpy as np
import os
import shutil
from slugify import slugify
from tqdm import tqdm

from utils import extract_emails_and_passwords


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
            self.cache[email].add(password)

    def persist(self):
        self._finalize_cache()
        with open(self.filename + '_per_user.json', 'w') as w:
            json.dump(fp=w, obj=self.cache_key_edit_distance_keep_user_struct, indent=4, sort_keys=True)

        sep = ' ||| '
        for edit_distance in sorted(self.cache_key_edit_distance_list):
            def csv_line_format(x):
                return str(edit_distance) + sep + x[0] + sep + x[1] + '\n'

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
    try:
        print('OUTPUT FOLDER: {0}.'.format(output_folder))
        shutil.rmtree(output_folder)
    except:
        pass
    os.makedirs(callback_output_dir)

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

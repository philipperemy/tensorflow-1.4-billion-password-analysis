import json

import editdistance
import numpy as np
import os

from shp import find_shortest_hamiltonian_path_in_complete_graph


class Callback:
    def __init__(self):
        pass

    def call(self, emails_passwords):
        # emails_passwords = list of tuples
        pass


class ReducePasswordsOnSimilarEmailsCallback(Callback):
    def __init__(self, persisted_filename):
        super().__init__()
        self.cache = {}
        self.cache_key_edit_distance_keep_user_struct = {}
        self.cache_key_edit_distance_list = {}
        self.filename = persisted_filename

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

            output_dir = os.path.join(os.path.expanduser('~/BreachCompilationAnalysis'), 'edit-distances')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            csv_file = os.path.join(output_dir, str(edit_distance) + '.csv')
            # print('Updating: ' + csv_file)
            with open(csv_file, encoding='utf8', mode='a') as w:
                password_pairs = self.cache_key_edit_distance_list[edit_distance]
                lines = list(map(csv_line_format, password_pairs))
                w.writelines(lines)

import itertools
import json
from itertools import combinations

import editdistance
import numpy as np

PASSWORDS = ['hello1', 'hello22', 'h@llo22', 'h@llo223']


def find_shortest_hamiltonian_path_in_complete_graph(passwords, debug=True):
    map_edit_distance = {}

    # shortest hamiltonian path in complete graph. NP-complete.
    for combo in combinations(passwords, 2):  # 2 for pairs, 3 for triplets, etc
        ed = editdistance.eval(combo[0], combo[1])
        if debug:
            print(combo[0], combo[1], ed)
        map_edit_distance[(combo[0], combo[1])] = ed
        map_edit_distance[(combo[1], combo[0])] = ed

    # factorial(n)
    permutations = list(itertools.permutations(passwords))

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


if __name__ == '__main__':
    print(find_shortest_hamiltonian_path_in_complete_graph(PASSWORDS, False))

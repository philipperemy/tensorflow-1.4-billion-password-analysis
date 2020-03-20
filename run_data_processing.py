import argparse

import os

from processing_callbacks import ReducePasswordsOnSimilarEmailsCallback
from utils import process

parser = argparse.ArgumentParser('Data Processing Tool.')
parser.add_argument('--breach_compilation_folder', type=os.path.expanduser,
                    help='BreachCompilation/ folder containing the 1.4 billion passwords dataset.', required=True)
parser.add_argument('--output_folder', type=os.path.expanduser,
                    default='~/BreachCompilationAnalysis',
                    help='Output folder containing the generated datasets.')
parser.add_argument('--max_num_files', type=int,
                    help='Maximum number of files to read. The entire dataset contains around 2000 files.'
                         'Can be useful to create mini datasets for the models.')


# ------------------------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------------------------
# INPUT: BreachCompilation/
# BreachCompilation is organized as:
#
# a/          - folder of emails starting with a
# a/a         - file of emails starting with aa
# a/b
# a/d
# ...
# z/
# ...
# z/y
# z/z
# ------------------------------------------------------------------------------
# OUTPUT: - BreachCompilationAnalysis/edit-distance/1.csv
#         - BreachCompilationAnalysis/edit-distance/2.csv
#         - BreachCompilationAnalysis/edit-distance/3.csv
#         [...]
#         > cat 1.csv
#             1 ||| samsung94 ||| samsung94@
#             1 ||| 040384alexej ||| 040384alexey
#             1 ||| HoiHalloDoeii14 ||| hoiHalloDoeii14
#             1 ||| hoiHalloDoeii14 ||| hoiHalloDoeii13
#             1 ||| hoiHalloDoeii13 ||| HoiHalloDoeii13
#             1 ||| 8znachnuu ||| 7znachnuu
#         EXPLANATION: edit-distance/ contains the passwords pairs sorted by edit distances.
#         1.csv contains all pairs with edit distance = 1 (exactly one addition, substitution or deletion).
#         2.csv => edit distance = 2, and so on.
#
#         - BreachCompilationAnalysis/ReducePasswordsOnSimilarEmailsCallback/99_per_user.json
#         - BreachCompilationAnalysis/ReducePasswordsOnSimilarEmailsCallback/9j_per_user.json
#         - BreachCompilationAnalysis/ReducePasswordsOnSimilarEmailsCallback/9a_per_user.json
#         [...]
#         > cat 96_per_user.json
#         {
#             "1.0": [
#             {
#                 "edit_distance": [
#                     0,
#                     1
#                 ],
#                 "email": "96-000@mail.ru",
#                 "password": [
#                     "090698d",
#                     "090698D"
#                 ]
#             },
#         {
#                 "edit_distance": [
#                     0,
#                     1
#                 ],
#                 "email": "96-96.1996@mail.ru",
#                 "password": [
#                     "5555555555q",
#                     "5555555555Q"
#                 ]
#          }
#         EXPLANATION: ReducePasswordsOnSimilarEmailsCallback/ contains files sorted by the first 2 letters of
#         the email address. For example 96-000@mail.ru will be located in 96_per_user.json
#         Each file lists all the passwords grouped by user and by edit distance.
#         For example, 96-000@mail.ru had 2 passwords: 090698d and 090698D. The edit distance between them is 1.
#         The edit_distance and the password arrays are of the same length, hence, a first 0 in the edit distance array.
#         Those files are useful to model how users change passwords over time.
#         We can't recover which one was the first password, but a shortest hamiltonian path algorithm is run
#         to detect the most probably password ordering for a user. For example:
#         hello => hello1 => hell@1 => hell@11 is the shortest path.
#         We assume that users are lazy by nature and that they prefer to change their password by the lowest number
#         of characters.


def run():
    # example: --breach_compilation_folder /media/philippe/DATA/BreachCompilation/
    # --max_num_files 100 --output_folder ~/BreachCompilationAnalysis2
    arg_p = parser.parse_args()
    process(breach_compilation_folder=arg_p.breach_compilation_folder,
            num_files=arg_p.max_num_files,
            output_folder=arg_p.output_folder,
            on_file_read_call_back_class=ReducePasswordsOnSimilarEmailsCallback)


if __name__ == '__main__':
    run()

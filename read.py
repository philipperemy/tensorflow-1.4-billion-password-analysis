import argparse

from callback import ReducePasswordsOnSimilarEmailsCallback
from utils import read_n_files

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--breach_compilation_folder',
                    type=str,
                    help='BreachCompilation/ containing the 1.4 billion passwords dataset.',
                    required=True)
parser.add_argument('--max_num_files',
                    type=int,
                    help='Maximum number of files to read. Not set means the entire dataset.')


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

def read():
    arg_p = parser.parse_args()
    read_n_files(breach_compilation_folder=arg_p.breach_compilation_folder,
                 num_files=arg_p.max_num_files,
                 on_file_read_call_back_class=ReducePasswordsOnSimilarEmailsCallback)


if __name__ == '__main__':
    read()

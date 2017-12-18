from callback import ReducePasswordsOnSimilarEmailsCallback
from utils import read_n_files

BREACH_COMPILATION_FOLDER = '/Users/philipperemy/Documents/BreachCompilation/'


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
    read_n_files(BREACH_COMPILATION_FOLDER, 1000, ReducePasswordsOnSimilarEmailsCallback)


if __name__ == '__main__':
    read()

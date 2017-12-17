import zip_pickle
from callback import ReducePasswordsOnSimilarEmailsCallback
from utils import read_n_files

BREACH_COMPILATION_FOLDER = '/Users/philipperemy/Documents/BreachCompilation/data'


def read():
    reduce_passwords_on_similar_emails = ReducePasswordsOnSimilarEmailsCallback()
    read_n_files(BREACH_COMPILATION_FOLDER, 100, reduce_passwords_on_similar_emails)
    persistence_filename = 'reduce_passwords_on_similar_emails.pkl.gzip'
    print('About to save {0} emails.'.format(len(reduce_passwords_on_similar_emails.cache)))
    zip_pickle.save(persistence_filename, reduce_passwords_on_similar_emails.cache)

    a = zip_pickle.load(persistence_filename)
    print(len(list(a)))


if __name__ == '__main__':
    read()

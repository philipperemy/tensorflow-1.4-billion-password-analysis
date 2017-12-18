import json

from callback import ReducePasswordsOnSimilarEmailsCallback
from utils import read_n_files

BREACH_COMPILATION_FOLDER = '/Users/philipperemy/Documents/BreachCompilation/data'


def read():
    reduce_passwords_on_similar_emails = ReducePasswordsOnSimilarEmailsCallback()
    read_n_files(BREACH_COMPILATION_FOLDER, 1000, reduce_passwords_on_similar_emails)
    print('About to save {0} emails.'.format(len(reduce_passwords_on_similar_emails.cache)))
    reduce_passwords_on_similar_emails.finalize_cache()
    with open('reduce_passwords_on_similar_emails.json', 'w') as w:
        json.dump(fp=w, obj=reduce_passwords_on_similar_emails.cache, indent=4)

        # zip_pickle.save(persistence_filename, reduce_passwords_on_similar_emails.cache)
        # a = zip_pickle.load(persistence_filename)


if __name__ == '__main__':
    read()

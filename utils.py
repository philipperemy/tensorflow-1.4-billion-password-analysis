from glob import glob

import numpy as np
import os
import shutil
from slugify import slugify
from tqdm import tqdm

from processing_callbacks import ReducePasswordsOnSimilarEmailsCallback

TMP_DIR = 'tmp'


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


def process(breach_compilation_folder,
            output_folder='~/BreachCompilationAnalysis',
            num_files=None,
            on_file_read_call_back_class=ReducePasswordsOnSimilarEmailsCallback):
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
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
    if num_files is not None:
        print('TRUNCATE DATASET TO: {0} files.'.format(num_files))
        all_filenames = all_filenames[0:num_files]

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


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = np.array([x for x in result if x is not None])
    pool.close()
    pool.join()
    return cleaned

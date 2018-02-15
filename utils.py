import multiprocessing
import os
import shutil
from functools import partial
from glob import glob

from slugify import slugify


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned


def extract_emails_and_passwords(txt_lines):
    emails_passwords = []
    for txt_line in txt_lines:
        try:
            if ':' in txt_line and '@' in txt_line:
                strip_txt_line = txt_line.strip()
                email, password = strip_txt_line.split(':')
                emails_passwords.append((email, password))
        except:
            pass
    return emails_passwords


def read_all(breach_compilation_folder, on_file_read_call_back):
    read_n_files(breach_compilation_folder, None, on_file_read_call_back)


def process_file(current_filename, output_dir, on_file_read_call_back_class):
    print(current_filename)
    if os.path.isfile(current_filename):
        suffix = slugify(current_filename.split('data')[-1])
        output_filename = os.path.join(output_dir, suffix)
        callback = on_file_read_call_back_class(output_filename)
        with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
            lines = r.readlines()
            emails_passwords = extract_emails_and_passwords(lines)
            callback.call(emails_passwords)
        print('Persisting {0} rows'.format(len(callback.cache)))
        print('Done for filename {0}'.format(current_filename))
        callback.persist()


def read_n_files(breach_compilation_folder, num_files, on_file_read_call_back_class):
    breach_compilation_folder = os.path.join(os.path.expanduser(breach_compilation_folder), 'data')
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
    all_filenames = list(filter(os.path.isfile, all_filenames))
    callback_class_name = str(on_file_read_call_back_class).split('callback.')[-1][:-2]
    output_dir = os.path.join(os.path.expanduser('~/BreachCompilationAnalysis'), callback_class_name)
    try:
        shutil.rmtree(output_dir)
    except:
        pass
    os.makedirs(output_dir)

    print('Output folder is {0}.'.format(output_dir))
    print('Found {0} files in {1}.'.format(len(all_filenames), breach_compilation_folder))
    if num_files is not None:
        all_filenames = all_filenames[0:num_files]

    num_threads = multiprocessing.cpu_count() // 2
    print('Going to run on {0} threads.'.format(num_threads))

    process_file_partial = partial(process_file,
                                   output_dir=output_dir,
                                   on_file_read_call_back_class=on_file_read_call_back_class)
    parallel_function(process_file_partial, all_filenames, num_threads)

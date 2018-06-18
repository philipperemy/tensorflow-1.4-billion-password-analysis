from glob import glob

import os
import shutil
from slugify import slugify
from tqdm import tqdm


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
        except:
            pass
    return emails_passwords


def process(breach_compilation_folder, num_files, on_file_read_call_back_class):
    breach_compilation_folder = os.path.join(os.path.expanduser(breach_compilation_folder), 'data')
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
    all_filenames = list(filter(os.path.isfile, all_filenames))
    callback_class_name = str(on_file_read_call_back_class).split('callback.')[-1][:-2]
    analysis_folder = os.path.expanduser('~/BreachCompilationAnalysis')
    output_dir = os.path.join(analysis_folder, callback_class_name)
    try:
        print('CLEAN FOLDER: {0}.'.format(analysis_folder))
        shutil.rmtree(analysis_folder)
    except:
        pass
    os.makedirs(output_dir)

    print('FOUND: {0} unique files in {1}.'.format(len(all_filenames), breach_compilation_folder))
    if num_files is not None:
        print('TRUNCATE DATASET TO: {0} files.'.format(num_files))
        all_filenames = all_filenames[0:num_files]

    bar = tqdm(all_filenames)
    for current_filename in bar:
        if os.path.isfile(current_filename):
            suffix = slugify(current_filename.split('data')[-1])
            output_filename = os.path.join(output_dir, suffix)
            callback = on_file_read_call_back_class(output_filename)
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
                emails_passwords = extract_emails_and_passwords(lines)
                callback.call(emails_passwords)
            bar.set_description('Processing {0} passwords'.format(len(callback.cache)))
            callback.persist()
    bar.close()
    print('DONE. SUCCESS.')
    print('OUTPUT: Dataset was generated at: {0}.'.format(analysis_folder))

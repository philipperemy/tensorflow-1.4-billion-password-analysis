import os
from glob import glob

from tqdm import tqdm


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


def read_n_files(breach_compilation_folder, num_files, on_file_read_call_back):
    all_filenames = glob(breach_compilation_folder + '/**/*', recursive=True)
    print('Found {0} files'.format(len(all_filenames)))
    if num_files is not None:
        all_filenames = all_filenames[0:num_files]
    for current_filename in tqdm(all_filenames):
        if os.path.isfile(current_filename):
            with open(current_filename, 'r', encoding='utf8', errors='ignore') as r:
                lines = r.readlines()
                emails_passwords = extract_emails_and_passwords(lines)
                on_file_read_call_back.call(emails_passwords)
                on_file_read_call_back.debug()

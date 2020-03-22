import click
import numpy as np
import os
import random


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


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = np.array([x for x in result if x is not None])
    pool.close()
    pool.join()
    return cleaned


class Ct:

    @staticmethod
    def input_file(writable=False):
        return click.Path(exists=True, file_okay=True, dir_okay=False,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def input_dir(writable=False):
        return click.Path(exists=True, file_okay=False, dir_okay=True,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def output_file():
        return click.Path(exists=False, file_okay=True, dir_okay=False,
                          writable=True, readable=True, resolve_path=True)

    @staticmethod
    def output_dir():
        return click.Path(exists=False, file_okay=False, dir_okay=True,
                          writable=True, readable=True, resolve_path=True)


def create_dir(output_dir: str):
    if len(output_dir) > 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def create_dir_for_file(filename: str):
    create_dir(os.path.dirname(filename))


def shuffle(lst):
    random.seed(123)
    random.shuffle(lst)

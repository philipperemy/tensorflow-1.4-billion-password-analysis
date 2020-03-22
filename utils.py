import click
import numpy as np
import os
import random


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


def recursive_help(cmd, parent=None):
    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    print(cmd.get_help(ctx))
    print()
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub, ctx)
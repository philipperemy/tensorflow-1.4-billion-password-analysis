import logging

import click

from core import preprocess, build_encodings, train
from utils import Ct, recursive_help


@click.group()
@click.option('--debug/--no-debug')
@click.pass_context
def cli(ctx, debug):
    if debug:
        click.echo('Debugging mode enabled.')
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)


@cli.command()
def dumphelp():
    recursive_help(cli)


@cli.command('preprocess')
@click.option('--breach_compilation_folder', required=True, type=Ct.input_dir())
@click.option('--output_dir', required=True, type=Ct.output_dir())
@click.option('--max_num_files', default=0, type=int, show_default=True)
def cli_preprocess(breach_compilation_folder, output_dir, max_num_files):
    preprocess(breach_compilation_folder, output_dir, max_num_files)


@cli.command('build-encodings')
@click.option('--edit_distance_file', required=True, type=Ct.input_file())
def cli_build_encodings(edit_distance_file):
    build_encodings(edit_distance_file)


@cli.command('train')
@click.option('--hidden_size', default=256, type=int, show_default=True)
@click.option('--batch_size', default=256, type=int, show_default=True)
def cli_train(hidden_size, batch_size):
    train(hidden_size, batch_size)


if __name__ == '__main__':
    cli()

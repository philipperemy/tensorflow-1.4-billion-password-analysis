import logging

import click

from core import preprocess, build_encodings, train
from utils import Ct


def recursive_help(cmd, parent=None):
    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    print(cmd.get_help(ctx))
    print()
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub, ctx)


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
@click.option('--output_folder', required=True, type=Ct.output_dir())
@click.option('--max_num_files', default=0, type=int, show_default=True)
def cli_preprocess(breach_compilation_folder, output_folder, max_num_files):
    preprocess(breach_compilation_folder, output_folder, max_num_files)


@cli.command('build-encodings')
@click.option('--training_filename', required=True, type=Ct.input_file())
def cli_build_encodings(training_filename):
    build_encodings(training_filename)


@cli.command('train')
@click.option('--hidden_size', default=256, type=int, show_default=True)
@click.option('--batch_size', default=256, type=int, show_default=True)
def cli_train(hidden_size, batch_size):
    return train(hidden_size, batch_size)


if __name__ == '__main__':
    cli()

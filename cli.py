import logging

import click

from core import preprocess
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
@click.option('--breach_compilation_folder', required=True, type=Ct.input_file())
@click.option('--output_folder', required=True, type=Ct.output_dir())
@click.option('--max_num_files', default=0, type=int, show_default=True)
def cli_preprocess(breach_compilation_folder, output_folder, max_num_files):
    preprocess(breach_compilation_folder, output_folder, max_num_files)


if __name__ == '__main__':
    cli()

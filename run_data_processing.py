import argparse

import os

from processing_callbacks import ReducePasswordsOnSimilarEmailsCallback
from utils import process


def get_script_arguments():
    parser = argparse.ArgumentParser('Data Processing Tool.')
    parser.add_argument('--breach_compilation_folder',
                        type=os.path.expanduser,
                        help='BreachCompilation/ folder containing the 1.4 billion passwords dataset.',
                        required=True)
    parser.add_argument('--output_folder', type=os.path.expanduser,
                        default='~/BreachCompilationAnalysis',
                        help='Output folder containing the generated datasets.')
    parser.add_argument('--max_num_files', type=int,
                        help='Maximum number of files to read. The entire dataset contains around 2000 files.'
                             'Can be useful to create mini datasets for the models.')
    return parser.parse_args()


def main():
    args = get_script_arguments()
    process(breach_compilation_folder=args.breach_compilation_folder,
            num_files=args.max_num_files,
            output_folder=args.output_folder,
            on_file_read_call_back_class=ReducePasswordsOnSimilarEmailsCallback)


if __name__ == '__main__':
    main()

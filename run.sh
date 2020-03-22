#!/usr/bin/env bash

set -e

if [[ $# -eq 0 ]] ; then
    echo 'Arguments: [breach_compilation_folder].'
    exit 0
fi

python cli.py preprocess --breach_compilation_folder $1 --max_num_files 100 --output_dir breach_compilation_preprocessed

python cli.py build-encodings --training_filename breach_compilation_preprocessed/edit-distances/1.csv

python cli.py train
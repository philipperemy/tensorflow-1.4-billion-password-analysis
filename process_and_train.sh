#!/usr/bin/env bash

if [[ $# -eq 0 ]] ; then
    echo 'USAGE: ./script.sh [breach_compilation_folder].'
    exit 0
fi

echo "Remove --max_num_files to process the entire dataset (few hours of processing in total)."

python run_data_processing.py --breach_compilation_folder $1 --max_num_files 100 --output_folder ~/BreachCompilationAnalysis

# Remove this.
rm -rf /tmp/indices_token.pkl /tmp/token_indices.pkl /tmp/x_y.npz
python run_encoding.py --training_filename ~/BreachCompilationAnalysis/edit-distances/1.csv

python train.py
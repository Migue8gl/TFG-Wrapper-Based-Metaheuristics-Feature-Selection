#!/bin/bash

# Optimizers
optimizers=('GOA' 'CS' 'GWO' 'FA')

# Datasets
datasets=(
    './datasets/spectf-heart.arff'
    './datasets/ionosphere.arff'
    './datasets/parkinsons.arff'
    './datasets/iris.arff'
    './datasets/wine.arff'
    './datasets/ecoli.arff'
    './datasets/breast-cancer.arff'
    './datasets/zoo.arff'
    './datasets/dermatology.arff'
    './datasets/sonar.arff'
)

# Define the file to store error messages
error_dir='results/logs'

# Create the file if it doesn't exist
if [ ! -d "$error_dir" ]; then
    mkdir -p "$error_dir"
fi

source env/bin/activate
./scripts/clean.sh -v False

# Run main.py for each optimizer
for opt in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset" | sed 's/\.arff$//')
        log_file="$error_dir/error_${opt}_${dataset_name}.log"
        touch "$log_file"
        python3 src/main.py -o "$opt" -d "$dataset" >"$log_file" 2>&1 &
    done
done

python3 scripts/monitor_threads.py -n True &




#!/bin/bash

# Optimizers
optimizers=('FA')

# Datasets
datasets=(
    './datasets/spectf-heart.arff'
    './datasets/ionosphere.arff'
    './datasets/parkinsons.arff'
    './datasets/iris.arff'
    './datasets/wine.arff'
    './datasets/ecoli.arff'
    './datasets/yeast.arff'
)

# Define the file to store error messages
error_dir='results/logs'

# Create the file if it doesn't exist
if [ ! -d "$error_dir" ]; then
    mkdir -p "$error_dir"
fi

source env/bin/activate

# Run main.py for each optimizer
for dataset in "${datasets[@]}"; do
    for opt in "${optimizers[@]}"; do
        dataset_name=$(basename "$dataset" | sed 's/\.arff$//')
        log_file="$error_dir/error_${opt}_${dataset_name}.log"
        touch "$log_file"
        python3 src/main.py -o "$opt" -d "$dataset" >"$log_file" 2>&1 &
    done
done

# Monitor the process threads
./scripts/monitor_process_threads.sh &


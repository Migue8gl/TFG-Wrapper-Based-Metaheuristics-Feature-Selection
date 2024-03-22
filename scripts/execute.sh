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

source env/bin/activate
./scripts/clean.sh -v False

# Define the file to store error messages
error_dir='results/logs'

# Create the file if it doesn't exist
mkdir -p "$error_dir"

# List of valid false values
TRUE_VALUES=("True" "true" "TRUE" "t" "T" "1")

# Check if the -v flag is set to false
if [ "$1" == "-n" ]; then
    for val in "${FALSE_VALUES[@]}"; do
        if [ "$2" == "$val" ]; then
            monitor_threads="python3 scripts/monitor_threads.py &"
        else
            monitor_threads=""
        fi
        break
    done
fi

for opt in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset" | sed 's/\.arff$//')
        log_file="$error_dir/error_${opt}_${dataset_name}.log"
        touch "$log_file"
        python3 src/main.py -o "$opt" -d "$dataset" ${1+"$@"} >"$log_file" 2>&1 &
    done
done

eval "$monitor_threads"

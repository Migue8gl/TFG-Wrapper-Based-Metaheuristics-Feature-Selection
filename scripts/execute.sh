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
for opt in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset" | sed 's/\.arff$//')
        log_file="$error_dir/error_${opt}_${dataset_name}.log"
        touch "$log_file"
        python3 src/main.py -o "$opt" -d "$dataset" >"$log_file" 2>&1 &
    done
done

# String to search for
process_name="python3 src/main.py"

# Script to execute when all threads are finished
python_script="scripts/group_csv.py"

while true; do
    # Check if there are process threads containing the string
    if pgrep -f "$process_name" > /dev/null; then
        :
    else
        # Execute the Python script
        python3 "$python_script"
        break  # Exit the while loop
    fi

    # Wait for 1 minute before checking again
    sleep 60
done




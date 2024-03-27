#!/bin/bash

#SBATCH -J wrapper_methods_fs
#SBATCH -p muylarga
#SBATCH ---array=0-263

# Optimizers
optimizers=('GWO' 'GOA' 'FA' 'CS')

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
    './datasets/yeast.arff'
)

source env/bin/activate
./scripts/clean.sh -v False

# Define the file to store error messages
error_dir='results/logs'

# Create the file if it doesn't exist
mkdir -p "$error_dir"

# Flag to track whether monitoring is required
monitor_required=false

# List of valid false values
TRUE_VALUES=("True" "true" "TRUE" "t" "T" "1")

# Check if the -n flag is set to True
if [ "$1" == "-n" ]; then
    for val in "${TRUE_VALUES[@]}"; do
        if [ "$2" == "$val" ]; then
            monitor_required=true
            break
        fi
    done
fi


for opt in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset" | sed 's/\.arff$//')
        log_file="$error_dir/error_${opt}_${dataset_name}.log"
        touch "$log_file"
        python3 src/main.py -o "$opt" -d "$dataset" ${1+"$@"} >"$log_file" 2>&1 &

        if [[ "$opt" != "aco" ]]; then
            log_file="$error_dir/error_${opt}_real_${dataset_name}.log"
            touch "$log_file"
            python3 src/main.py -o "$opt" -b 'r' -d "$dataset" ${1+"$@"} >"$log_file" 2>&1 &
        fi
    done
done



# Execute monitor_threads.py if monitoring is required
if [ "$monitor_required" = true ]; then
    python3 scripts/monitor_threads.py &
fi

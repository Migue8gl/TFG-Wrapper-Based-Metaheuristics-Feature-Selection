#!/bin/bash

# Define the directories
RESULTS_DIR="results/"
IMAGES_DIR="images/"
VERBOSE=true

# List of valid false values
FALSE_VALUES=("False" "false" "FALSE" "f" "F" "0")

# Check if the -v flag is set to false
if [ "$1" == "-v" ]; then
    for val in "${FALSE_VALUES[@]}"; do
        if [ "$2" == "$val" ]; then
            VERBOSE=false
            break
        fi
    done
fi

# Function to print messages if VERBOSE is true
print_message() {
    if [ "$VERBOSE" == true ]; then
        echo "$1"
    fi
}

# Check if RESULTS_DIR exists
if [ -d "$RESULTS_DIR" ]; then
    # Clean everything in the results directory except the root directory itself
    print_message "Cleaning $RESULTS_DIR directory..."
    find "$RESULTS_DIR" -mindepth 1 -delete
else
    print_message "$RESULTS_DIR does not exist."
fi

# Check if IMAGES_DIR exists
if [ -d "$IMAGES_DIR" ]; then
    # Clean everything in the images directory except the root directory itself
    print_message "Cleaning $IMAGES_DIR directory..."
    find "$IMAGES_DIR" -mindepth 1 -delete
else
    print_message "$IMAGES_DIR does not exist."
fi

print_message "Cleanup complete."

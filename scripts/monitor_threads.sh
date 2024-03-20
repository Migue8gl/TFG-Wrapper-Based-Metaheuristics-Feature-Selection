#!/bin/bash

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
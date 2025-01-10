#!/bin/bash

# Clean previous results
echo "Cleaning previous results..."
rm -f results/all_results.csv
rm -f results/best_hyperparams.json

# Run the master node
echo "Starting Master Node..."
python3 src/master.py

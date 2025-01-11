#!/bin/bash

if [ -d "results" ]; then
    echo "Cleaning results directory..."
    rm -f results/*.csv results/*.json
    echo "Results directory cleaned."
else
    echo "Results directory not found. Creating now..."
    mkdir results
fi

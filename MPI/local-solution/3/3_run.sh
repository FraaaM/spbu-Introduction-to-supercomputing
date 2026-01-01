#!/bin/bash

rm -f 3_results.csv

echo "Compiling..."
mpicxx 3.cpp -o ping_pong -O3

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Running Ping-Pong benchmark..."

mpirun --oversubscribe -np 2 ./ping_pong

echo "Done. Results saved in 3_results.csv"
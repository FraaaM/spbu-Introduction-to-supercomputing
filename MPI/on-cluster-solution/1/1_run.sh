#!/bin/bash

#SBATCH --nodes=8                  
#SBATCH --ntasks=128  
#SBATCH --ntasks-per-node=16            

module purge
module load gcc/10
module load openmpi

echo "Compiling..."
mpicxx 1.cpp -o mpi_min_finder -O3

rm -f 1_results.csv

TEST_PAIRS="1:1 1:2 1:4 1:8 1:16 2:4 2:8 2:16 2:32 4:8 4:16 4:32 4:64 8:16 8:32 8:64 8:128"

echo "Starting benchmarks..."

for PAIR in $TEST_PAIRS; do
    N=$(echo $PAIR | cut -d: -f1)
    P=$(echo $PAIR | cut -d: -f2)
    
    P_PER_NODE=$((P / N))

    echo "--- Running Nodes=$N, Total Procs=$P (Procs per Node=$P_PER_NODE) ---"
    
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_min_finder $N
done

echo "Tests finished. Results are in 1_results.csv"
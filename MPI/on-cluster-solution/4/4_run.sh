#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16

module purge
module load gcc/10
module load openmpi

echo "Compiling matrix multiplication program..."
mpicxx 4.cpp -o mpi_matrix -O3

rm -f 4_results.csv

# Алгоритм Фокса сработает только для P = 1, 4, 16, 64.
# Ленточный сработает для всех, где N кратно P.
TEST_PAIRS="1:1 1:2 1:4 1:8 1:16 2:4 2:8 2:16 2:32 4:16 4:32 4:64 8:64 8:128"

echo "Starting benchmarks..."

for PAIR in $TEST_PAIRS; do
    N=$(echo $PAIR | cut -d: -f1)
    P=$(echo $PAIR | cut -d: -f2)
    
    P_PER_NODE=$((P / N))

    echo "--- Running Nodes=$N, Total Procs=$P ---"
    
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_matrix $N
done

echo "Tests finished. Results in 4_results.csv"
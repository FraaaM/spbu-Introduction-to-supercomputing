#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8

module purge
module load gcc/10
module load openmpi

echo "Compiling..."
mpicxx 6.cpp -o mpi_modes -O3

rm -f 6_results.csv

# для алгоритма Фокса нужны квадратные числа процессов:
# 1, 4, 16, 64...

TEST_PAIRS="1:1 1:4 2:16 4:16 4:64 8:64"

echo "Starting communication modes benchmarks..."

for PAIR in $TEST_PAIRS; do
    N=$(echo $PAIR | cut -d: -f1)
    P=$(echo $PAIR | cut -d: -f2)
    P_PER_NODE=$((P / N))

    echo "--- Running Nodes=$N, Total Procs=$P ---"
    
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_modes $N
done

echo "Tests finished. Results in 6_results.csv"
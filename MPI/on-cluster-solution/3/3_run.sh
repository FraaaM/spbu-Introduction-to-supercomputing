#!/bin/bash

#SBATCH --nodes=2                 
#SBATCH --ntasks=2                

module purge
module load gcc/10
module load openmpi

echo "Compiling ping-pong test..."
mpicxx 3.cpp -o ping_pong -O3

rm -f 3_results.csv

TEST_PAIRS="1:2 2:2"

echo "Starting Ping-Pong benchmarks..."

for PAIR in $TEST_PAIRS; do
    N=$(echo $PAIR | cut -d: -f1)
    P=$(echo $PAIR | cut -d: -f2) 
    
    P_PER_NODE=$((P / N))

    echo "--- Running Nodes=$N, Total Procs=$P (Procs per Node=$P_PER_NODE) ---"
    
    # Запуск
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./ping_pong $N
done

echo "Tests finished. Results are in 3_results.csv"
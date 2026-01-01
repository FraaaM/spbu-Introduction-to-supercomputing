#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16

module purge
module load gcc/10
module load openmpi

echo "Compiling..."
mpicxx 5.cpp -o mpi_balance -O3

rm -f 5_results.csv

TEST_PAIRS="1:1 1:2 1:4 1:8 1:16 2:4 2:8 2:16 2:32 4:8 4:16 4:32 4:64 8:16 8:32 8:64 8:128"

#  СЦЕНАРИИ 
# 1. COMPUTE BOUND 
# Ожидание: Линейное ускорение (время падает пропорционально числу процессов).
SCENARIO_1_NAME="ComputeHeavy"
SCENARIO_1_WORK=1000000000 # 1 млрд итераций 
SCENARIO_1_MSG=1            # 1 байт

# 2. LATENCY BOUND 
# Ожидание: Плохое масштабирование, сетевые задержки съедят выгоду.
SCENARIO_2_NAME="LatencyBound"
SCENARIO_2_WORK=10000000    # 10 млн итераций 
SCENARIO_2_MSG=100            # 100 байт

# 3. BANDWIDTH BOUND 
# Ожидание: Упрется в пропускную способность сети при выходе за пределы 1 узла.
SCENARIO_3_NAME="BandwidthBound"
SCENARIO_3_WORK=10000000  # 10 млн итераций
SCENARIO_3_MSG=1048576     # 1 МБ

# 4. MAX BANDWIDTH BOUND 
# Ожидание: Упрется в пропускную способность сети при выходе за пределы 1 узла.
SCENARIO_4_NAME="MaxBandwidthBound"
SCENARIO_4_WORK=1000000  # 1 млн итераций
SCENARIO_4_MSG=10485760     # 10 МБ

echo "Starting benchmarks..."

for PAIR in $TEST_PAIRS; do
    N=$(echo $PAIR | cut -d: -f1)
    P=$(echo $PAIR | cut -d: -f2)
    P_PER_NODE=$((P / N))

    echo "--- Config: Nodes=$N, Procs=$P ---"

    # 1. COMPUTE BOUND
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_balance $SCENARIO_1_NAME $SCENARIO_1_WORK $SCENARIO_1_MSG $N
    # 2. LATENCY BOUND
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_balance $SCENARIO_2_NAME $SCENARIO_2_WORK $SCENARIO_2_MSG $N
    # 3. BANDWIDTH BOUND
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_balance $SCENARIO_3_NAME $SCENARIO_3_WORK $SCENARIO_3_MSG $N
    # 4. MAX BANDWIDTH BOUND
    srun --nodes=$N --ntasks=$P --ntasks-per-node=$P_PER_NODE ./mpi_balance $SCENARIO_4_NAME $SCENARIO_4_WORK $SCENARIO_4_MSG $N

done

echo "Done. See 5_results.csv"
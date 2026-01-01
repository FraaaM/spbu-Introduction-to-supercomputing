#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

// Функция имитации нагрузки
void do_heavy_work(long long iterations) {
    volatile double result = 0.0;
    for (long long i = 0; i < iterations; ++i) {
        result += std::sin(i) * std::cos(i);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) std::cerr << "Usage: ./prog <ScenarioName> <TotalWorkIter> <MsgSizeBytes>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string scenario_name = argv[1];
    long long total_work = std::atoll(argv[2]); 
    int msg_size = std::atoi(argv[3]);          
    
    int num_nodes = 1;
    if (argc > 4) num_nodes = std::atoi(argv[4]);

    const std::string OUTPUT_FILE = "7_results.csv";
    const int STEPS = 50; 

    long long local_work = total_work / size;
    long long work_per_step = local_work / STEPS;

    std::vector<char> send_buf(msg_size, 'x');
    std::vector<char> recv_buf(msg_size);

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    // BLOCKING
    MPI_Barrier(MPI_COMM_WORLD);
    double start_block = MPI_Wtime();

    for (int step = 0; step < STEPS; ++step) {
        do_heavy_work(work_per_step);
        MPI_Sendrecv(send_buf.data(), msg_size, MPI_CHAR, right, 0,
                     recv_buf.data(), msg_size, MPI_CHAR, left, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double end_block = MPI_Wtime();
    double local_time_block = end_block - start_block;
    double global_time_block = 0.0;
    MPI_Reduce(&local_time_block, &global_time_block, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // NON-BLOCKING (Overlap)
    MPI_Barrier(MPI_COMM_WORLD);
    double start_nonblock = MPI_Wtime();

    MPI_Request reqs[2];
    for (int step = 0; step < STEPS; ++step) {
        MPI_Irecv(recv_buf.data(), msg_size, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(send_buf.data(), msg_size, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[1]);
        
        do_heavy_work(work_per_step);
        
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    double end_nonblock = MPI_Wtime();
    double local_time_nonblock = end_nonblock - start_nonblock;
    double global_time_nonblock = 0.0;
    MPI_Reduce(&local_time_nonblock, &global_time_nonblock, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ifstream check_file(OUTPUT_FILE);
        bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
        check_file.close();

        std::ofstream out(OUTPUT_FILE, std::ios::app);
        if (write_header) {
            out << "Scenario,Method,Nodes,Processes,TotalWork,MsgBytes,Time_s\n";
        }
        
        out << scenario_name << ",Blocking," << num_nodes << "," << size << "," 
            << total_work << "," << msg_size << "," << global_time_block << "\n";
            
        out << scenario_name << ",NonBlocking," << num_nodes << "," << size << "," 
            << total_work << "," << msg_size << "," << global_time_nonblock << "\n";

        std::cout << "[" << scenario_name << "] P=" << size << " | Block: " << global_time_block 
                  << "s | NonBlock: " << global_time_nonblock << "s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Error: Ping-Pong requires exactly 2 processes. You started " << size << "." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int num_nodes = 0;
    if (argc > 1) {
        num_nodes = std::atoi(argv[1]);
    }

    const std::string OUTPUT_FILE = "3_results.csv";
    
    std::vector<int> message_sizes;
    for (int s = 1; s <= 4 * 1024 * 1024; s *= 2) {
        message_sizes.push_back(s);
    }
    const int REPETITIONS = 1000; 

    if (rank == 0) {
        std::ifstream check_file(OUTPUT_FILE);
        bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
        check_file.close();

        if (write_header) {
            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << "Nodes,Processes,Bytes,Time_sec,Bandwidth_MBs\n";
        }
    }


    std::vector<char> buffer(4 * 1024 * 1024, 'a');

    for (int n : message_sizes) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        double t_start = MPI_Wtime();

        for (int i = 0; i < REPETITIONS; ++i) {
            if (rank == 0) {
                MPI_Send(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        double t_end = MPI_Wtime();
        
        if (rank == 0) {
            double total_time = t_end - t_start;
            double time_per_msg = total_time / (2.0 * REPETITIONS);
            double bandwidth = (static_cast<double>(n) / (1024.0 * 1024.0)) / time_per_msg;

            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << num_nodes << "," << size << "," << n << "," 
                << std::scientific << std::setprecision(9) << time_per_msg << "," 
                << std::fixed << std::setprecision(4) << bandwidth << "\n";
            
            if (n == 1024*1024) { 
                std::cout << "Nodes: " << num_nodes << " | 1MB Bandwidth: " << bandwidth << " MB/s" << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // нужно  2 процесса
    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Error: This program requires exactly 2 processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }


    std::vector<int> message_sizes;
    for (int s = 1; s <= 4 * 1024 * 1024; s *= 2) {
        message_sizes.push_back(s);
    }

    const int REPETITIONS = 1000; 
    const std::string OUTPUT_FILE = "3_results.csv";

    if (rank == 0) {
        std::cout << "Starting Ping-Pong test with " << REPETITIONS << " repetitions..." << std::endl;
        std::ofstream out(OUTPUT_FILE);
        out << "Bytes,Time_sec,Bandwidth_MBs\n";
        out.close();
    }

    for (int n : message_sizes) {
        std::vector<char> buffer(n, 'a');

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
        double total_time = t_end - t_start;

        if (rank == 0) {
            double time_per_msg = total_time / (2.0 * REPETITIONS);
            
            double bandwidth = (static_cast<double>(n) / (1024.0 * 1024.0)) / time_per_msg;

            std::cout << "Size: " << std::setw(8) << n << " bytes | "
                      << "Time: " << std::scientific << std::setprecision(3) << time_per_msg << " s | "
                      << "Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " MB/s" << std::endl;

            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << n << "," << std::scientific << std::setprecision(9) << time_per_msg << "," 
                << std::fixed << std::setprecision(4) << bandwidth << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
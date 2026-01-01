#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <fstream>
#include <string>

void fill_vector(std::vector<int>& vec) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 100000000);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

int get_local_min(const std::vector<int>& local_data) {
    int local_min = std::numeric_limits<int>::max();
    for (int x : local_data) {
        if (x < local_min) {
            local_min = x;
        }
    }
    return local_min;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_nodes = 1;
    if (argc > 1) {
        num_nodes = std::atoi(argv[1]);
    }

    const std::vector<int> TEST_SIZES = { 1000000, 10000000, 100000000 };
    const std::string OUTPUT_FILE = "1_results.csv";

    for (int N : TEST_SIZES) {
        std::vector<int> data;
        std::vector<int> local_data;
        int global_min = 0;
        double elapsed_time = 0.0;

        //MPI_Scatter требует равных частей. 
        int chunk_size = N / size;
        
        if (rank == 0) {
            data.resize(chunk_size * size);
            fill_vector(data);
        }

        local_data.resize(chunk_size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Scatter(data.data(), chunk_size, MPI_INT, 
                    local_data.data(), chunk_size, MPI_INT, 
                    0, MPI_COMM_WORLD);

        int local_min = get_local_min(local_data);

        MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        elapsed_time = MPI_Wtime() - t0;

        if (rank == 0) {
            std::ifstream check_file(OUTPUT_FILE);
            bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
            check_file.close();

            std::ofstream out(OUTPUT_FILE, std::ios::app);
            if (out.is_open()) {
                if (write_header) {
                    out << "Nodes,Processes,VectorSize,GlobalMin,Time_s\n";
                }
                out << num_nodes << "," << size << "," << N << "," << global_min << "," << elapsed_time << "\n";
                out.close();
            } else {
                std::cerr << "Error opening output file on rank 0." << std::endl;
            }

            std::cout << "Test Size=" << N << ", Nodes=" << num_nodes 
                      << ", Procs=" << size << ", Time=" << elapsed_time 
                      << "s, Min=" << global_min << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
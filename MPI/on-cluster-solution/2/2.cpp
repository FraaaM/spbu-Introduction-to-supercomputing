#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>

void fill_vector(std::vector<double>& vec) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

double get_local_scalar(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
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
    const std::string OUTPUT_FILE = "2_results.csv";

    for (int N : TEST_SIZES) {
        std::vector<double> v1, v2;
        std::vector<double> local_v1, local_v2;
        
        double global_sum = 0.0;
        double elapsed_time = 0.0;

        int chunk_size = N / size;

        if (rank == 0) {
            v1.resize(chunk_size * size);
            v2.resize(chunk_size * size);
            
            fill_vector(v1);
            fill_vector(v2);
        }

        local_v1.resize(chunk_size);
        local_v2.resize(chunk_size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Scatter(v1.data(), chunk_size, MPI_DOUBLE, 
                    local_v1.data(), chunk_size, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        MPI_Scatter(v2.data(), chunk_size, MPI_DOUBLE, 
                    local_v2.data(), chunk_size, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        double local_sum = get_local_scalar(local_v1, local_v2);

        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        elapsed_time = MPI_Wtime() - t0;

        if (rank == 0) {
            std::ifstream check_file(OUTPUT_FILE);
            bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
            check_file.close();

            std::ofstream out(OUTPUT_FILE, std::ios::app);
            if (out.is_open()) {
                if (write_header) {
                    out << "Nodes,Processes,VectorSize,ScalarProduct,Time_s\n";
                }
                out << num_nodes << "," << size << "," << N << "," << global_sum << "," << elapsed_time << "\n";
                out.close();
            } else {
                std::cerr << "Error opening output file on rank 0." << std::endl;
            }

            std::cout << "Test Size=" << N << ", Nodes=" << num_nodes 
                      << ", Procs=" << size << ", Time=" << elapsed_time 
                      << "s, Result=" << global_sum << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

const std::vector<int> TEST_SIZES = { 512, 1024, 2048 }; 

const std::string OUTPUT_FILE = "4_results.csv";

inline int idx(int r, int c, int cols) {
    return r * cols + c;
}

void fill_matrix(std::vector<double>& mat, int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (auto& val : mat) val = dist(rng);
}

// Ленточное разбиение
// A разбивается по строкам, B дублируется на всех узлах
double run_striped(int N, int rank, int size, std::vector<double>& A, std::vector<double>& B) {
    if (N % size != 0) return -1.0; 

    int local_rows = N / size;
    std::vector<double> local_A(local_rows * N);
    std::vector<double> local_C(local_rows * N, 0.0);

    if (rank != 0) B.resize(N * N);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A.data(), local_rows * N, MPI_DOUBLE, 
                local_A.data(), local_rows * N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // IJK умножение
    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < N; ++k) {
            double temp = local_A[idx(i, k, N)];
            for (int j = 0; j < N; ++j) {
                local_C[idx(i, j, N)] += temp * B[idx(k, j, N)];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    return MPI_Wtime() - start;
}

// 2. АЛГОРИТМ ФОКСА (Блочное разбиение)
// Требует квадратного количества процессов (grid_size * grid_size == size)
double run_fox(int N, int rank, int size, const std::vector<double>& A_flat, const std::vector<double>& B_flat) {
    int q = (int)std::sqrt(size);
    if (q * q != size) return -1.0; 

    if (N % q != 0) return -2.0; 

    int block_dim = N / q; 
    int block_size = block_dim * block_dim;

    // Создание декартовой решетки
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; 
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

    int grid_coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, grid_coords);
    int row = grid_coords[0];
    int col = grid_coords[1];

    // Локальные блоки
    std::vector<double> local_A(block_size);
    std::vector<double> local_B(block_size);
    std::vector<double> local_C(block_size, 0.0);
    std::vector<double> temp_A(block_size); 

    // Распределение данных
    if (rank == 0) {
        std::vector<double> send_buffer_A(block_size);
        std::vector<double> send_buffer_B(block_size);
        
        for(int i=0; i<block_dim; ++i) {
            for(int j=0; j<block_dim; ++j) {
                local_A[i*block_dim + j] = A_flat[i*N + j];
                local_B[i*block_dim + j] = B_flat[i*N + j];
            }
        }

        // Рассылка остальным
        for (int p = 1; p < size; ++p) {
            int p_coords[2];
            MPI_Cart_coords(grid_comm, p, 2, p_coords);
            int r_start = p_coords[0] * block_dim;
            int c_start = p_coords[1] * block_dim;

            // Упаковка блока для процесса P
            for(int i=0; i<block_dim; ++i) {
                for(int j=0; j<block_dim; ++j) {
                    send_buffer_A[i*block_dim + j] = A_flat[(r_start + i)*N + (c_start + j)];
                    send_buffer_B[i*block_dim + j] = B_flat[(r_start + i)*N + (c_start + j)];
                }
            }
            MPI_Send(send_buffer_A.data(), block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(send_buffer_B.data(), block_size, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_A.data(), block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B.data(), block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Подготовка коммуникаторов для строк и столбцов
    int sub_dims[2];
    sub_dims[0] = 0; sub_dims[1] = 1; // Оставляем Y (строка)
    MPI_Comm row_comm;
    MPI_Cart_sub(grid_comm, sub_dims, &row_comm);

    sub_dims[0] = 1; sub_dims[1] = 0; // Оставляем X (столбец)
    int source, dest;
    MPI_Cart_shift(grid_comm, 0, -1, &source, &dest); // -1 значит сдвиг вверх (индекс уменьшается)

    // ОСНОВНОЙ ЦИКЛ ФОКСА 
    for (int stage = 0; stage < q; ++stage) {
        // 1. Выбор корня для broadcast в строке: (row + stage) % q
        int bcast_root = (row + stage) % q;

        if (col == bcast_root) {
            temp_A = local_A;
        }

        MPI_Bcast(temp_A.data(), block_size, MPI_DOUBLE, bcast_root, row_comm);

        // Локальное умножение: C += temp_A * local_B
        for (int i = 0; i < block_dim; ++i) {
            for (int k = 0; k < block_dim; ++k) {
                double ta = temp_A[i * block_dim + k];
                for (int j = 0; j < block_dim; ++j) {
                    local_C[i * block_dim + j] += ta * local_B[k * block_dim + j];
                }
            }
        }

        // Циклический сдвиг B вверх
        MPI_Sendrecv_replace(local_B.data(), block_size, MPI_DOUBLE, 
                             dest, 0, source, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid_comm);

    return elapsed;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_nodes = 1;
    if (argc > 1) num_nodes = std::atoi(argv[1]);

    if (rank == 0) {
        std::ifstream check_file(OUTPUT_FILE);
        bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
        check_file.close();

        if (write_header) {
            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << "Nodes,Processes,MatrixSize,Method,Time_s\n";
        }
    }

    for (int N : TEST_SIZES) {
        std::vector<double> A, B;

        if (rank == 0) {
            A.resize(N * N);
            B.resize(N * N);
            fill_matrix(A, N);
            fill_matrix(B, N);
        }

        //  Запуск Striped (Ленточный)
        // Требует N % size == 0 
        double time_striped = -1.0;
        if (N % size == 0) {
            time_striped = run_striped(N, rank, size, A, B);
        }

        if (rank == 0 && time_striped > 0) {
            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << num_nodes << "," << size << "," << N << ",Striped," << time_striped << "\n";
            std::cout << "P=" << size << " N=" << N << " Method=Striped Time=" << time_striped << std::endl;
        }

        // Запуск Fox (Блочный)
        // Требует size = q*q
        double time_fox = run_fox(N, rank, size, A, B);
        
        if (rank == 0) {
            std::ofstream out(OUTPUT_FILE, std::ios::app);
            if (time_fox > 0) {
                out << num_nodes << "," << size << "," << N << ",Fox," << time_fox << "\n";
                std::cout << "P=" << size << " N=" << N << " Method=Fox Time=" << time_fox << std::endl;
            } else if (time_fox == -1.0) {
                // out << num_nodes << "," << size << "," << N << ",Fox,N/A_NotSquare\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
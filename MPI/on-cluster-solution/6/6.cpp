#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <map>

const std::vector<int> TEST_SIZES = {512, 1024, 2048}; 
const std::string OUTPUT_FILE = "6_results.csv";

// Режимы коммуникации
enum CommMode { MODE_STANDARD, MODE_SYNC, MODE_BUFFERED };

inline int idx(int r, int c, int cols) { return r * cols + c; }

void fill_matrix(std::vector<double>& mat, int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (auto& val : mat) val = dist(rng);
}

// АЛГОРИТМ ФОКСА (модифицированный для задачи 6)
// Принимает параметр mode для выбора типа отправки
double run_fox(int N, int rank, int size, const std::vector<double>& A_flat, const std::vector<double>& B_flat, CommMode mode) {
    int q = (int)std::sqrt(size);
    if (q * q != size) return -1.0; 
    if (N % q != 0) return -2.0; 

    int block_dim = N / q; 
    int block_size = block_dim * block_dim;

    // Настройка коммуникатора 
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
    std::vector<double> recv_B_buffer(block_size); 

    // Подготовка буфера для MPI_Bsend 
    std::vector<char> bsend_buffer; 
    if (mode == MODE_BUFFERED) {
        int msg_size = block_size * sizeof(double);
        int buf_size = msg_size + (MPI_BSEND_OVERHEAD * 10) + 409600; 
        bsend_buffer.resize(buf_size);
        MPI_Buffer_attach(bsend_buffer.data(), buf_size);
    }

    //  РАСПРЕДЕЛЕНИЕ ДАННЫХ  
    if (rank == 0) {
        std::vector<double> s_A(block_size), s_B(block_size);
        for(int i=0; i<block_dim; ++i) {
            for(int j=0; j<block_dim; ++j) {
                local_A[i*block_dim + j] = A_flat[i*N + j];
                local_B[i*block_dim + j] = B_flat[i*N + j];
            }
        }
        // Рассылка
        for (int p = 1; p < size; ++p) {
            int p_c[2]; MPI_Cart_coords(grid_comm, p, 2, p_c);
            int rs = p_c[0] * block_dim;
            int cs = p_c[1] * block_dim;
            for(int i=0; i<block_dim; ++i) {
                for(int j=0; j<block_dim; ++j) {
                    s_A[i*block_dim + j] = A_flat[(rs + i)*N + (cs + j)];
                    s_B[i*block_dim + j] = B_flat[(rs + i)*N + (cs + j)];
                }
            }
            MPI_Send(s_A.data(), block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(s_B.data(), block_size, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_A.data(), block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B.data(), block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Коммуникаторы для строк и столбцов
    int sub_dims[2] = {0, 1}; // Y
    MPI_Comm row_comm;
    MPI_Cart_sub(grid_comm, sub_dims, &row_comm);

    int source, dest; // Сдвиг по столбцу (X)
    MPI_Cart_shift(grid_comm, 0, -1, &source, &dest); 

    //  ОСНОВНОЙ ЦИКЛ ФОКСА 
    for (int stage = 0; stage < q; ++stage) {
        // 1. Broadcast A (всегда Standard Bcast, т.к. коллективная операция)
        int bcast_root = (row + stage) % q;
        if (col == bcast_root) temp_A = local_A;
        MPI_Bcast(temp_A.data(), block_size, MPI_DOUBLE, bcast_root, row_comm);

        // 2. Умножение
        for (int i = 0; i < block_dim; ++i) {
            for (int k = 0; k < block_dim; ++k) {
                double ta = temp_A[i * block_dim + k];
                for (int j = 0; j < block_dim; ++j) {
                    local_C[i * block_dim + j] += ta * local_B[k * block_dim + j];
                }
            }
        }

        // 3. Сдвиг B (здесь реализованы разные режимы)
        // Чтобы избежать дэдлоков в Synchronous режиме, сначала запускается Irecv
        MPI_Request req_recv;
        MPI_Irecv(recv_B_buffer.data(), block_size, MPI_DOUBLE, source, 0, grid_comm, &req_recv);

        // отправка
        if (mode == MODE_STANDARD) {
            MPI_Send(local_B.data(), block_size, MPI_DOUBLE, dest, 0, grid_comm);
        } 
        else if (mode == MODE_SYNC) {
            MPI_Ssend(local_B.data(), block_size, MPI_DOUBLE, dest, 0, grid_comm);
        } 
        else if (mode == MODE_BUFFERED) {
            MPI_Bsend(local_B.data(), block_size, MPI_DOUBLE, dest, 0, grid_comm);
        }

        MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        local_B = recv_B_buffer;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    // Очистка буфера для Bsend
    if (mode == MODE_BUFFERED) {
        void* buffer_addr;
        int buffer_size;
        MPI_Buffer_detach(&buffer_addr, &buffer_size);
    }
    
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

    // Карта имен режимов для вывода
    std::map<CommMode, std::string> mode_names = {
        {MODE_STANDARD, "Standard"},
        {MODE_SYNC, "Synchronous"},
        {MODE_BUFFERED, "Buffered"}
    };

    if (rank == 0) {
        std::ifstream check_file(OUTPUT_FILE);
        bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
        check_file.close();

        if (write_header) {
            std::ofstream out(OUTPUT_FILE, std::ios::app);
            out << "Nodes,Processes,MatrixSize,Mode,Time_s\n";
        }
    }

    int q = (int)std::sqrt(size);
    bool is_square = (q * q == size);

    if (is_square) {
        for (int N : TEST_SIZES) {
            std::vector<double> A, B;

            if (rank == 0) {
                A.resize(N * N);
                B.resize(N * N);
                fill_matrix(A, N);
                fill_matrix(B, N);
            }

            // Запуск каждого режима
            for (auto const& [mode, name] : mode_names) {
                
                // Синхронизация между тестами режимов
                MPI_Barrier(MPI_COMM_WORLD);

                double t = run_fox(N, rank, size, A, B, mode);

                if (rank == 0) {
                    if (t > 0) {
                        std::ofstream out(OUTPUT_FILE, std::ios::app);
                        out << num_nodes << "," << size << "," << N << "," << name << "," << t << "\n";
                        std::cout << "P=" << size << " N=" << N << " Mode=" << name << " Time=" << t << "s" << std::endl;
                    } else {
                        std::cerr << "Error running Fox for N=" << N << std::endl;
                    }
                }
            }
        }
    } else {
        if (rank == 0) {
            std::cout << "Skipping Fox tests: processes (" << size << ") is not a perfect square." << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
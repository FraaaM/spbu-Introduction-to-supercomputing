#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

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
    long long total_work = std::atoll(argv[2]); // Общий объем работы на ВСЕХ
    int msg_size = std::atoi(argv[3]);          // Размер сообщения
    
    // кол-во узлов (передается скриптом 4-м параметром)
    int num_nodes = 1;
    if (argc > 4) num_nodes = std::atoi(argv[4]);

    const std::string OUTPUT_FILE = "5_results.csv";
    const int COMMUNICATION_STEPS = 50; // Количество обменов данными 

    long long local_work = total_work / size;

    std::vector<char> send_buf(msg_size, 'x');
    std::vector<char> recv_buf(msg_size);

    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int step = 0; step < COMMUNICATION_STEPS; ++step) {
        
        // Вычисления (делим работу на шаги, чтобы перемешивать с коммуникацией)
        do_heavy_work(local_work / COMMUNICATION_STEPS);

        // Коммуникация (SendRecv - безопасный обмен по кольцу)
        MPI_Sendrecv(send_buf.data(), msg_size, MPI_CHAR, right_neighbor, 0,
                     recv_buf.data(), msg_size, MPI_CHAR, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;

    // Находим максимальное время среди всех
    double global_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ifstream check_file(OUTPUT_FILE);
        bool write_header = check_file.peek() == std::ifstream::traits_type::eof();
        check_file.close();

        std::ofstream out(OUTPUT_FILE, std::ios::app);
        if (write_header) {
            out << "Scenario,Nodes,Processes,TotalWork,MsgBytes,Time_s\n";
        }
        
        out << scenario_name << "," << num_nodes << "," << size << "," 
            << total_work << "," << msg_size << "," << global_elapsed << "\n";

        std::cout << "[" << scenario_name << "] P=" << size << " Time=" << global_elapsed << "s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
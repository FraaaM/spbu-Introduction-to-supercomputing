#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <numeric>

// Размер вектора (10 млн элементов)
const size_t TEST_SIZE = 10000000;

// Потоки
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

const std::string OUTPUT_FILE = "..\\src_results/7.csv";

void fill_vector(std::vector<double>& vec) {
    std::fill(vec.begin(), vec.end(), 1.0);
}

// 1. Встроенная редукция 
double run_omp_reduction(const std::vector<double>& data, int num_threads, double& result_out) {
    double total_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+:total_sum) num_threads(num_threads)
    for (size_t i = 0; i < data.size(); ++i) {
        total_sum += data[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = total_sum;
    return std::chrono::duration<double>(end - start).count();
}

// 2. Atomic (Атомарные операции)
double run_atomic(const std::vector<double>& data, int num_threads, double& result_out) {
    double total_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;
#pragma omp for nowait
        for (size_t i = 0; i < data.size(); ++i) {
            local_sum += data[i];

#pragma omp atomic
            total_sum += local_sum;

        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = total_sum;
    return std::chrono::duration<double>(end - start).count();
}

// 3. Critical 
double run_critical(const std::vector<double>& data, int num_threads, double& result_out) {
    double total_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;
#pragma omp for nowait
        for (size_t i = 0; i < data.size(); ++i) {
            local_sum += data[i];


            // Critical: полностью блокирует выполнение блока для других потоков
#pragma omp critical
            {
                total_sum += local_sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = total_sum;
    return std::chrono::duration<double>(end - start).count();
}

// 4. Locks (Замки)
double run_locks(const std::vector<double>& data, int num_threads, double& result_out) {
    double total_sum = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock); // Инициализация замка

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;
#pragma omp for nowait
        for (size_t i = 0; i < data.size(); ++i) {
            local_sum += data[i];


            // Ручная установка и снятие замка
            omp_set_lock(&lock);
            total_sum += local_sum;
            omp_unset_lock(&lock);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    omp_destroy_lock(&lock); // Удаление замка

    result_out = total_sum;
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    namespace fs = std::filesystem;
    fs::path output_path(OUTPUT_FILE);
    if (output_path.has_parent_path() && !fs::exists(output_path.parent_path())) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream csv_file(OUTPUT_FILE);
    if (!csv_file.is_open()) return 1;

    csv_file << "Method,Size,Threads,Time,Result" << std::endl;

    std::cout << "Generating data (Size: " << TEST_SIZE << ")..." << std::endl;
    std::vector<double> data(TEST_SIZE);
    fill_vector(data);

    std::cout << "Starting Synchronization benchmarks..." << std::endl;

    for (int threads : TEST_THREADS) {
        std::cout << "Threads: " << threads << "..." << std::flush;
        double res = 0;
        double time = 0;

        // 1. OMP Reduction
        time = run_omp_reduction(data, threads, res);
        csv_file << "OMP_Reduction," << TEST_SIZE << "," << threads << "," << time << "," << res << std::endl;

        // 2. Atomic
        time = run_atomic(data, threads, res);
        csv_file << "Atomic," << TEST_SIZE << "," << threads << "," << time << "," << res << std::endl;

        // 3. Critical
        time = run_critical(data, threads, res);
        csv_file << "Critical," << TEST_SIZE << "," << threads << "," << time << "," << res << std::endl;

        // 4. Locks
        time = run_locks(data, threads, res);
        csv_file << "Locks," << TEST_SIZE << "," << threads << "," << time << "," << res << std::endl;

        std::cout << " Done." << std::endl;
    }

    csv_file.close();
    std::cout << "Results saved to " << OUTPUT_FILE << std::endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <limits>
#include <fstream>
#include <string>
#include <filesystem>

// Размеры массивов для тестирования (1M, 10M, 100M)
const std::vector<int> TEST_SIZES = { 1000000, 10000000, 100000000 };

// Количество потоков 
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

// Путь к результату
const std::string OUTPUT_FILE = "..\\src_results/1.csv";


// Функция для заполнения вектора случайными числами
void fill_vector(std::vector<int>& vec) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 100000000);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

// Без редукции 
double run_no_reduction(const std::vector<int>& data, int num_threads, int& result_out) {
    int min_val = std::numeric_limits<int>::max();
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
    {
        int local_min = std::numeric_limits<int>::max();

#pragma omp for nowait 
        for (int i = 0; i < data.size(); ++i) {
            if (data[i] < local_min) {
                local_min = data[i];
            }
        }

#pragma omp critical
        {
            if (local_min < min_val) {
                min_val = local_min;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = min_val;
    return std::chrono::duration<double>(end - start).count();
}

// С редукцией
double run_with_reduction(const std::vector<int>& data, int num_threads, int& result_out) {
    int min_val = std::numeric_limits<int>::max();

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(min:min_val) num_threads(num_threads)
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = min_val;
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    namespace fs = std::filesystem;
    fs::path output_path(OUTPUT_FILE);
    if (output_path.has_parent_path() && !fs::exists(output_path.parent_path())) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream csv_file(OUTPUT_FILE);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open output file " << OUTPUT_FILE << std::endl;
        return 1;
    }

    csv_file << "Method,Size,Threads,Time,Result" << std::endl;

    std::cout << "Starting benchmarks..." << std::endl;
    std::cout << "Results will be saved to: " << OUTPUT_FILE << std::endl;

    for (int size : TEST_SIZES) {
        std::cout << "\nGenerating data for size: " << size << "..." << std::endl;
        std::vector<int> data(size);
        fill_vector(data);

        for (int threads : TEST_THREADS) {
            std::cout << "  Running with " << threads << " threads..." << std::flush;

            int res_no_red = 0;
            int res_red = 0;

            double time_no_red = run_no_reduction(data, threads, res_no_red);
            csv_file << "NoReduction," << size << "," << threads << "," << time_no_red << "," << res_no_red << std::endl;

            double time_red = run_with_reduction(data, threads, res_red);
            csv_file << "Reduction," << size << "," << threads << "," << time_red << "," << res_red << std::endl;

            if (res_no_red != res_red) {
                std::cerr << " [ERROR: Results differ!]";
            }
            std::cout << " Done." << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nAll benchmarks completed." << std::endl;

    return 0;
}
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

// 1000x1000 = 1M элементов
// 5000x5000 = 25M элементов
// 10000x10000 = 100M элементов
const std::vector<size_t> TEST_DIMS = { 1000, 5000, 10000,};
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };
const std::string OUTPUT_FILE = "..\\src_results/4.csv";

// Функция генерации матрицы
void generate_matrix(std::vector<std::vector<int>>& matrix, size_t size) {
    std::mt19937 rng(42); 
    std::uniform_int_distribution<int> dist(-10000, 10000);

    matrix.assign(size, std::vector<int>(size));

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
}

// Основная функция вычисления (Max of Mins)
double run_max_of_mins(const std::vector<std::vector<int>>& matrix, int num_threads, int& result_out) {
    int max_of_mins = std::numeric_limits<int>::min();

    auto start = std::chrono::high_resolution_clock::now();

    // Распараллеливание внешниего цика (по строкам).
    // reduction(max:max_of_mins) собирает максимальное значение со всех потоков.
    omp_set_num_threads(num_threads);
#pragma omp parallel for reduction(max:max_of_mins)
    for (int i = 0; i < matrix.size(); ++i) {

        // Поиск минимума в конкретной строке (последовательно)
        int min_in_row = std::numeric_limits<int>::max();
        for (int val : matrix[i]) {
            if (val < min_in_row) {
                min_in_row = val;
            }
        }

        if (min_in_row > max_of_mins) {
            max_of_mins = min_in_row;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    result_out = max_of_mins;
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

    std::cout << "Starting MaxOfMins benchmarks..." << std::endl;
    std::cout << "Results will be saved to: " << OUTPUT_FILE << std::endl;

    std::vector<std::vector<int>> matrix;

    for (size_t dim : TEST_DIMS) {
        size_t total_elements = dim * dim;
        std::cout << "\nGenerating matrix " << dim << "x" << dim << " (" << total_elements << " elements)..." << std::endl;

        generate_matrix(matrix, dim);

        for (int threads : TEST_THREADS) {
            std::cout << "  Running with " << threads << " threads..." << std::flush;

            int result = 0;
            double time_seconds = run_max_of_mins(matrix, threads, result);

            csv_file << "MaxOfMins," << total_elements << "," << threads << "," << time_seconds << "," << result << std::endl;

            std::cout << " Done. Time: " << time_seconds << "s" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nAll benchmarks completed." << std::endl;

    return 0;
}
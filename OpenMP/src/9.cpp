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

// Размеры матриц (NxN)
const std::vector<size_t> TEST_SIZES = { 1000, 10000, 20000 };

// Потоки
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

const std::string OUTPUT_FILE = "..\\src_results/9.csv";

// Генерация
void generate_matrix(std::vector<std::vector<int>>& matrix, size_t size) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    matrix.assign(size, std::vector<int>(size));
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            matrix[i][j] = dist(rng);
}

// Функция решения
// mode 1: Только внешний (Outer)
// mode 2: Только внутренний (Inner)
// mode 3: Вложенный (Nested)
double run_test(const std::vector<std::vector<int>>& matrix, int num_threads, int mode, int& result_out) {
    int max_of_mins = std::numeric_limits<int>::min();
    int rows = (int)matrix.size();
    int cols = (int)matrix[0].size();

    auto start = std::chrono::high_resolution_clock::now();

    // Outer Only (Классический)
    if (mode == 1) {
        omp_set_nested(0); // Выключаем вложенность
        omp_set_num_threads(num_threads);

#pragma omp parallel for reduction(max:max_of_mins)
        for (int i = 0; i < rows; ++i) {
            int min_in_row = std::numeric_limits<int>::max();
            for (int j = 0; j < cols; ++j) {
                if (matrix[i][j] < min_in_row) min_in_row = matrix[i][j];
            }
            if (min_in_row > max_of_mins) max_of_mins = min_in_row;
        }
    }
    // Inner Only (Тяжелый)
    else if (mode == 2) {
        omp_set_nested(0);
        omp_set_num_threads(num_threads);

        for (int i = 0; i < rows; ++i) {
            int min_in_row = std::numeric_limits<int>::max();

            // Параллелим внутренний. Потоки создаются заново на каждой строке.
#pragma omp parallel for reduction(min:min_in_row)
            for (int j = 0; j < cols; ++j) {
                if (matrix[i][j] < min_in_row) min_in_row = matrix[i][j];
            }

            if (min_in_row > max_of_mins) max_of_mins = min_in_row;
        }
    }
    
    // Nested (Вложенный)
    else if (mode == 3) {
        omp_set_nested(1); // Включаем вложенность
        omp_set_num_threads(num_threads); 

#pragma omp parallel for reduction(max:max_of_mins)
        for (int i = 0; i < rows; ++i) {
            int min_in_row = std::numeric_limits<int>::max();

            // потоков: num_threads * 2
#pragma omp parallel for reduction(min:min_in_row) num_threads(2)
            for (int j = 0; j < cols; ++j) {
                if (matrix[i][j] < min_in_row) min_in_row = matrix[i][j];
            }

            if (min_in_row > max_of_mins) max_of_mins = min_in_row;
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
    if (!csv_file.is_open()) return 1;

    csv_file << "Strategy,Size,Threads,Time,Result" << std::endl;

    // Проверка поддержки 
    std::cout << "Default Nested Status: " << omp_get_nested() << std::endl;
    omp_set_nested(1);
    std::cout << "Nested Status after Enabling: " << omp_get_nested() << std::endl;

    std::cout << "Starting benchmarks..." << std::endl;
    std::vector<std::vector<int>> matrix;

    for (size_t size : TEST_SIZES) {
        std::cout << "\nSize: " << size << "x" << size << "..." << std::endl;
        generate_matrix(matrix, size);
        int res = 0;
        double t = 0;

        //  OUTER
        std::cout << "  [Outer]..." << std::flush;
        for (int threads : TEST_THREADS) {
            t = run_test(matrix, threads, 1, res);
            csv_file << "Outer," << size << "," << threads << "," << t << "," << res << std::endl;
        }
        std::cout << " Done." << std::endl;

        // INNER
        std::cout << "  [Inner]..." << std::flush;
        for (int threads : TEST_THREADS) {
            if (size > 2000 && threads > 4) continue;

            t = run_test(matrix, threads, 2, res);
            csv_file << "Inner," << size << "," << threads << "," << t << "," << res << std::endl;
        }
        std::cout << " Done." << std::endl;

        // NESTED
        std::cout << "  [Nested]..." << std::flush;
        for (int threads : TEST_THREADS) {
            t = run_test(matrix, threads, 3, res);
            csv_file << "Nested," << size << "," << threads << "," << t << "," << res << std::endl;
        }
        std::cout << " Done." << std::endl;
    }

    csv_file.close();
    std::cout << "Saved to " << OUTPUT_FILE << std::endl;
    return 0;
}
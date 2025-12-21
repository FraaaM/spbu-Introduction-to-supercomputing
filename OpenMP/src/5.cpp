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
#include <map>

// Размеры матриц (количество строк)
const std::vector<size_t> TEST_SIZES = { 1000, 5000, 10000, 50000 };

// Потоки
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

// Типы планировщиков
enum class SchedType { Static, Dynamic, Guided };
const std::vector<std::pair<SchedType, std::string>> SCHEDULES = {
    {SchedType::Static, "Static"},
    {SchedType::Dynamic, "Dynamic"},
    {SchedType::Guided, "Guided"}
};

// Типы матриц
enum class MatrixType { Triangular, Band };
const std::vector<std::pair<MatrixType, std::string>> MATRIX_TYPES = {
    {MatrixType::Triangular, "Triangular"},
    {MatrixType::Band, "Band"}
};

const std::string OUTPUT_FILE = "..\\src_results/5.csv";

// Генерация треугольной матрицы (нижней)
size_t generate_triangular(std::vector<std::vector<int>>& matrix, size_t rows) {
    matrix.resize(rows);
    size_t count = 0;

    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < static_cast<int>(rows); ++i) {
        matrix[i].resize(i + 1); 
        count += (i + 1);
        for (auto& val : matrix[i]) val = 1; 
    }
    return count;
}

// Генерация ленточной матрицы
// Храним только элементы внутри ленты.
// Ширина полуленты (bandwidth) = rows / 5 (20%)
size_t generate_band(std::vector<std::vector<int>>& matrix, size_t rows) {
    matrix.resize(rows);
    int half_width = rows / 5;
    if (half_width < 1) half_width = 1;

    size_t count = 0;

    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < static_cast<int>(rows); ++i) {
        int start_col = std::max(0, i - half_width);
        int end_col = std::min((int)rows, i + half_width + 1);

        int row_len = end_col - start_col;

        matrix[i].resize(row_len);
        count += row_len;
        for (auto& val : matrix[i]) val = 1;
    }
    return count;
}

// Функция вычисления
double run_max_of_mins(const std::vector<std::vector<int>>& matrix, int num_threads, SchedType sched_type, int& result_out) {
    int max_of_mins = std::numeric_limits<int>::min();
    int chunk_size = 100; // Размер пачки итераций

    switch (sched_type) {
    case SchedType::Static:  omp_set_schedule(omp_sched_static, 0); break;
    case SchedType::Dynamic: omp_set_schedule(omp_sched_dynamic, chunk_size); break;
    case SchedType::Guided:  omp_set_schedule(omp_sched_guided, 0); break;
    }

    auto start = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(runtime) reduction(max:max_of_mins)
    for (int i = 0; i < matrix.size(); ++i) {
        int min_in_row = std::numeric_limits<int>::max();
        for (int val : matrix[i]) {
            if (val < min_in_row) min_in_row = val;
        }
        if (min_in_row > max_of_mins) max_of_mins = min_in_row;
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

    csv_file << "MatrixType,Schedule,Size,Threads,Time,Result" << std::endl;

    std::cout << "Starting Matrix benchmarks..." << std::endl;
    std::vector<std::vector<int>> matrix;

    for (const auto& mat_pair : MATRIX_TYPES) {
        MatrixType m_type = mat_pair.first;
        std::string m_name = mat_pair.second;

        std::cout << "\n=== Matrix Type: " << m_name << " ===" << std::endl;

        for (size_t rows : TEST_SIZES) {
            size_t total_elements = 0;

            if (m_type == MatrixType::Triangular) {
                total_elements = generate_triangular(matrix, rows);
            }
            else {
                total_elements = generate_band(matrix, rows);
            }

            std::cout << "  Size: " << rows << " rows (" << total_elements << " elements)..." << std::endl;

            for (const auto& sched_pair : SCHEDULES) {
                SchedType s_type = sched_pair.first;
                std::string s_name = sched_pair.second;

                for (int threads : TEST_THREADS) {
                    int result = 0;
                    double time_s = run_max_of_mins(matrix, threads, s_type, result);

                    csv_file << m_name << "," << s_name << "," << total_elements << ","
                        << threads << "," << time_s << "," << result << std::endl;
                }
            }
        }
    }

    csv_file.close();
    std::cout << "\nBenchmarks completed." << std::endl;
    return 0;
}
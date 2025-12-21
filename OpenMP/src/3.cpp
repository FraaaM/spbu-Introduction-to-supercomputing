#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <string>

// Значения N (количество шагов интегрирования) 
const std::vector<size_t> TEST_SIZES = {100000, 1000000, 10000000, 100000000 };

// Количество потоков 
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

const std::string OUTPUT_FILE = "..\\src_results/3.csv";

// Функция, которую интегрируем: sin(x)
double f(double x) {
    return std::sin(x);
}

// Функция вычисления интеграла (Метод средних прямоугольников)
double run_integral(double a, double b, size_t n, int num_threads, double& result_out) {
    double h = (b - a) / static_cast<double>(n);
    double total_sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+:total_sum) num_threads(num_threads)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        // x_i - середина отрезка
        double x_i = a + (i + 0.5) * h;
        total_sum += f(x_i);
    }

    auto end = std::chrono::high_resolution_clock::now();

    result_out = total_sum * h;
    return std::chrono::duration<double>(end - start).count(); // Время в секундах
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

    std::cout << "Starting Integration benchmarks..." << std::endl;
    std::cout << "Results will be saved to: " << OUTPUT_FILE << std::endl;

    const double a = 0.0;
    const double b = 3.14159265358979323846; // Интеграл синуса от 0 до PI равен 2.0

    for (size_t n : TEST_SIZES) {
        std::cout << "\nRunning for N steps: " << n << "..." << std::endl;

        for (int threads : TEST_THREADS) {
            std::cout << "  Threads: " << threads << "..." << std::flush;

            double result = 0.0;
            double time_seconds = run_integral(a, b, n, threads, result);

            csv_file << "Rectangle method," << n << "," << threads << "," << time_seconds << "," << result << std::endl;

            std::cout << " Done. Time: " << time_seconds << "s" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nAll benchmarks completed." << std::endl;

    return 0;
}
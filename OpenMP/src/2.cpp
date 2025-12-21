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
#include <cmath>

const std::vector<size_t> TEST_SIZES = { 1000000, 10000000, 100000000 };

const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

const std::string OUTPUT_FILE = "..\\src_results/2.csv";

// Функция для заполнения вектора случайными числами
void fill_vector(std::vector<double>& vec) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

// Без редукции 
double run_no_reduction(const std::vector<double>& v1, const std::vector<double>& v2, int num_threads, double& result_out) {
    double total_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
    {
        double local_sum = 0.0;

#pragma omp for nowait 
        for (size_t i = 0; i < v1.size(); ++i) {
            local_sum += v1[i] * v2[i];
        }

#pragma omp critical
        {
            total_sum += local_sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result_out = total_sum;
    return std::chrono::duration<double>(end - start).count();
}

// С редукцией 
double run_with_reduction(const std::vector<double>& v1, const std::vector<double>& v2, int num_threads, double& result_out) {
    double total_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    // reduction(+:total_sum) автоматически создает локальные копии и суммирует их в конце
#pragma omp parallel for reduction(+:total_sum) num_threads(num_threads)
    for (size_t i = 0; i < v1.size(); ++i) {
        total_sum += v1[i] * v2[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
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
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open output file " << OUTPUT_FILE << std::endl;
        return 1;
    }

    csv_file << "Method,Size,Threads,Time,Result" << std::endl;

    std::cout << "Starting Scalar Product benchmarks..." << std::endl;
    std::cout << "Results will be saved to: " << OUTPUT_FILE << std::endl;

    for (size_t size : TEST_SIZES) {
        std::cout << "\nGenerating data for size: " << size << "..." << std::endl;
        std::vector<double> v1(size);
        std::vector<double> v2(size);
        fill_vector(v1);
        fill_vector(v2);

        for (int threads : TEST_THREADS) {
            std::cout << "  Running with " << threads << " threads..." << std::flush;

            double res_no_red = 0.0;
            double res_red = 0.0;

            double time_no_red = run_no_reduction(v1, v2, threads, res_no_red);
            csv_file << "NoReduction," << size << "," << threads << "," << time_no_red << "," << res_no_red << std::endl;

            double time_red = run_with_reduction(v1, v2, threads, res_red);
            csv_file << "Reduction," << size << "," << threads << "," << time_red << "," << res_red << std::endl;

           /* if (std::abs(res_no_red - res_red) > 1e-5) {
                std::cerr << " [ERROR: Results differ! " << res_no_red << " vs " << res_red << "]";
            }*/
            std::cout << " Done." << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nAll benchmarks completed." << std::endl;

    return 0;
}
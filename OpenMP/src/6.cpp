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

// Размеры задачи (количество итераций цикла)
const std::vector<size_t> TEST_SIZES = { 10000, 100000, 1000000 };

// Количество потоков
const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

// Типы планировщиков
enum class SchedType { Static, Dynamic, Guided };
const std::vector<std::pair<SchedType, std::string>> SCHEDULES = {
    {SchedType::Static, "Static"},
    {SchedType::Dynamic, "Dynamic"},
    {SchedType::Guided, "Guided"}
};

const std::string OUTPUT_FILE = "..\\src_results/6.csv";

// Функция для генерации "карты сложности"
// 90% итераций будут легкими, 10% - очень тяжелыми.
void generate_workload(std::vector<int>& workload, size_t size) {
    workload.resize(size);
    std::mt19937 rng(42);

    std::uniform_int_distribution<int> light_dist(10, 100);
    std::uniform_int_distribution<int> heavy_dist(10000, 50000);
    std::uniform_int_distribution<int> chance_dist(0, 99);

    for (size_t i = 0; i < size; ++i) {
        if (chance_dist(rng) < 10) {
            workload[i] = heavy_dist(rng);
        }
        else {
            workload[i] = light_dist(rng);
        }
    }
}

// Имитация тяжелой работы
// Выполняет ops вычислений синуса
double heavy_computation(int ops) {
    double res = 0.0;
    for (int i = 0; i < ops; ++i) {
        res += std::sin(i * 0.1);
    }
    return res;
}

double run_experiment(const std::vector<int>& workload, int num_threads, SchedType sched_type, double& result_out) {
    double total_sum = 0.0;
    int chunk_size = 50;

    switch (sched_type) {
    case SchedType::Static:
        omp_set_schedule(omp_sched_static, 0);
        break;
    case SchedType::Dynamic:
        omp_set_schedule(omp_sched_dynamic, chunk_size);
        break;
    case SchedType::Guided:
        omp_set_schedule(omp_sched_guided, 0);
        break;
    }

    auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);

    // schedule(runtime) берет настройки из omp_set_schedule
    #pragma omp parallel for schedule(runtime) reduction(+:total_sum)
    for (size_t i = 0; i < workload.size(); ++i) {
        total_sum += heavy_computation(workload[i]);
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
    if (!csv_file.is_open()) return 1;

    csv_file << "WorkloadType,Schedule,Size,Threads,Time,Result" << std::endl;

    std::cout << "Starting Unbalanced Loop benchmarks..." << std::endl;
    std::vector<int> workload;

    for (size_t size : TEST_SIZES) {
        std::cout << "\nGenerating workload for size " << size << "..." << std::endl;
        generate_workload(workload, size);

        for (const auto& sched_pair : SCHEDULES) {
            std::cout << "  Testing " << sched_pair.second << "..." << std::flush;

            for (int threads : TEST_THREADS) {
                double result = 0.0;
                double time = run_experiment(workload, threads, sched_pair.first, result);

                csv_file << "RandomImbalance," << sched_pair.second << "," << size << ","
                    << threads << "," << time << "," << result << std::endl;
            }
            std::cout << " Done." << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nBenchmarks completed." << std::endl;
    return 0;
}
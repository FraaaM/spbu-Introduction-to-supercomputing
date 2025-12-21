#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <mutex>

const std::string OUTPUT_FILE = "..\\src_results/8.csv";

// должны быть те же конфигурации, что и в генераторе
const std::vector<std::pair<int, int>> TEST_CONFIGS = { 
    {1000, 100}, 
    {1000, 1000}, 
    {2000, 2000} 
};

const std::vector<int> TEST_THREADS = { 1, 2, 4, 8, 12, 16, 24, 32 };

double run_sections(const std::string& filename, int num_threads, double& result_out) {
    omp_set_nested(1);
    omp_set_num_threads(num_threads);

    std::vector<std::vector<double>> buffer;
    std::mutex mtx;
    std::atomic<bool> finished{ false };
    double total_sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel sections
    {
        // SECTION 1: PRODUCER (Чтение) 
        #pragma omp section
        {
            std::ifstream ifs(filename);
            if (ifs.is_open()) {
                int M, N;
                if (ifs >> M >> N) {
                    for (int i = 0; i < M * 2; ++i) {
                        std::vector<double> vec(N);
                        for (int j = 0; j < N; ++j) {
                            ifs >> vec[j];
                        }

                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            buffer.push_back(std::move(vec));
                        }
                    }
                }
            }
            finished = true;
        }

        // SECTION 2: CONSUMER (Счет) 
        #pragma omp section
        {
            while (!finished || !buffer.empty()) {
                std::vector<std::vector<double>> local_copy;

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (!buffer.empty()) {
                        local_copy.swap(buffer);
                    }
                }

                if (!local_copy.empty()) {
                    double local_batch_sum = 0.0;
                    int dim = local_copy[0].size();
                    int batch_size = (int)local_copy.size();

                    // Если пачка большая, параллелим внутри Consumer
                    // идем с шагом 2, так как в файле пары: (VecA, VecB), (VecA, VecB)...
                    #pragma omp parallel for reduction(+:local_batch_sum)
                    for (int i = 0; i < batch_size - 1; i += 2) {
                        double dot = 0.0;
                        for (int k = 0; k < dim; ++k) {
                            dot += local_copy[i][k] * local_copy[i + 1][k];
                        }
                        local_batch_sum += dot;
                    }
                    total_sum += local_batch_sum;
                }
                else {
                    std::this_thread::yield();
                }
            }
        }
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
        std::cerr << "Error: cannot open output file" << std::endl;
        return 1;
    }

    csv_file << "Config,Threads,Time,Result" << std::endl;
    std::cout << "Starting Sections benchmark (Separate Files)..." << std::endl;

    for (auto& p : TEST_CONFIGS) {
        int pairs = p.first;
        int dim = p.second;
        
        std::string filename = "vectors_" + std::to_string(pairs) + "_" + std::to_string(dim) + ".txt";
        
        if (!fs::exists(filename)) {
            std::cerr << "[WARNING] File " << filename << " not found! Run gen_vectors.py first.\n";
            continue;
        }

        std::string config_name = std::to_string(pairs) + "x" + std::to_string(dim);
        std::cout << "\nRunning Config: " << config_name << std::endl;

        for (int threads : TEST_THREADS) {
            std::cout << "  Threads: " << threads << "..." << std::flush;
            
            double res = 0.0;
            double time = run_sections(filename, threads, res);

            csv_file << config_name << "," << threads << "," << time << "," << res << std::endl;
            std::cout << " Done. Time: " << time << "s" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Done." << std::endl;
    return 0;
}
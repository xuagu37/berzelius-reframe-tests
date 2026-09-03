#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << operation << " failed with cuBLAS status "
                << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(operation) check_cuda((operation), #operation)
#define CUBLAS_CHECK(operation) check_cublas((operation), #operation)

struct Options {
    int gemm_n = 16384;
    int gemm_warmup = 100;
    int gemm_samples = 50;
    std::size_t hbm_mib = 1024;
    int hbm_warmup = 20;
    int hbm_samples = 50;
    int hbm_kernels_per_sample = 20;
    std::string output = "pilot_samples.csv";
};

int parse_positive_int(const std::string& value, const std::string& name) {
    std::size_t parsed = 0;
    const long result = std::stol(value, &parsed);
    if (parsed != value.size() || result <= 0 || result > INT32_MAX) {
        throw std::invalid_argument(name + " must be a positive integer");
    }

    return static_cast<int>(result);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto separator = argument.find('=');
        if (separator == std::string::npos || argument.rfind("--", 0) != 0) {
            throw std::invalid_argument("expected --name=value, got " + argument);
        }

        const std::string name = argument.substr(2, separator - 2);
        const std::string value = argument.substr(separator + 1);
        if (name == "gemm-n") {
            options.gemm_n = parse_positive_int(value, name);
        } else if (name == "gemm-warmup") {
            options.gemm_warmup = parse_positive_int(value, name);
        } else if (name == "gemm-samples") {
            options.gemm_samples = parse_positive_int(value, name);
        } else if (name == "hbm-mib") {
            options.hbm_mib = static_cast<std::size_t>(
                parse_positive_int(value, name)
            );
        } else if (name == "hbm-warmup") {
            options.hbm_warmup = parse_positive_int(value, name);
        } else if (name == "hbm-samples") {
            options.hbm_samples = parse_positive_int(value, name);
        } else if (name == "hbm-kernels-per-sample") {
            options.hbm_kernels_per_sample = parse_positive_int(value, name);
        } else if (name == "output") {
            if (value.empty()) {
                throw std::invalid_argument("output must not be empty");
            }
            options.output = value;
        } else {
            throw std::invalid_argument("unknown option --" + name);
        }
    }

    if (options.gemm_n % 8 != 0) {
        throw std::invalid_argument("gemm-n must be a multiple of 8");
    }

    return options;
}

__global__ void fill_half(__half* values, std::size_t count, __half value) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        values[index] = value;
    }
}

__global__ void fill_float(float* values, std::size_t count, float value) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        values[index] = value;
    }
}

__global__ void triad(
    const float* first,
    const float* second,
    float* result,
    std::size_t count,
    float scalar
) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        result[index] = first[index] + scalar * second[index];
    }
}

template<typename Operation>
std::vector<double> measure_samples(
    int samples,
    int operations_per_sample,
    Operation operation
) {
    std::vector<cudaEvent_t> starts(samples);
    std::vector<cudaEvent_t> stops(samples);
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventCreate(&starts[sample]));
        CUDA_CHECK(cudaEventCreate(&stops[sample]));
    }

    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(starts[sample]));
        for (int operation_index = 0;
             operation_index < operations_per_sample;
             ++operation_index) {
            operation();
        }
        CUDA_CHECK(cudaEventRecord(stops[sample]));
    }
    CUDA_CHECK(cudaEventSynchronize(stops.back()));

    std::vector<double> milliseconds(samples);
    for (int sample = 0; sample < samples; ++sample) {
        float elapsed = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, starts[sample], stops[sample]));
        milliseconds[sample] = elapsed / operations_per_sample;
        CUDA_CHECK(cudaEventDestroy(starts[sample]));
        CUDA_CHECK(cudaEventDestroy(stops[sample]));
    }

    return milliseconds;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        throw std::invalid_argument("cannot calculate an empty percentile");
    }

    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double coefficient_of_variation_percent(const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double squared_difference_sum = 0.0;
    for (const double value : values) {
        const double difference = value - mean;
        squared_difference_sum += difference * difference;
    }

    const double variance = values.size() > 1
        ? squared_difference_sum / static_cast<double>(values.size() - 1)
        : 0.0;
    return 100.0 * std::sqrt(variance) / mean;
}

std::string gpu_uuid(const cudaUUID_t& uuid) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
        if (index == 3 || index == 5 || index == 7 || index == 9) {
            output << '-';
        }
    }
    return output.str();
}

std::string hostname() {
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0) {
        throw std::runtime_error("gethostname failed");
    }
    return buffer;
}

std::string environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? "unset" : value;
}

struct BenchmarkResult {
    std::vector<double> milliseconds;
    std::vector<double> rates;
    double first_value;
    double last_value;
    double expected_value;
};

BenchmarkResult run_gemm(const Options& options) {
    const std::size_t element_count =
        static_cast<std::size_t>(options.gemm_n) * options.gemm_n;
    const std::size_t matrix_bytes = element_count * sizeof(__half);
    __half* first = nullptr;
    __half* second = nullptr;
    __half* result = nullptr;
    CUDA_CHECK(cudaMalloc(&first, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&second, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&result, matrix_bytes));

    constexpr int threads = 256;
    constexpr int blocks = 4096;
    const __half input_value = __float2half(1.0F / 128.0F);
    fill_half<<<blocks, threads>>>(first, element_count, input_value);
    fill_half<<<blocks, threads>>>(second, element_count, input_value);
    CUDA_CHECK(cudaGetLastError());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto gemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            options.gemm_n,
            options.gemm_n,
            options.gemm_n,
            &alpha,
            first,
            CUDA_R_16F,
            options.gemm_n,
            second,
            CUDA_R_16F,
            options.gemm_n,
            &beta,
            result,
            CUDA_R_16F,
            options.gemm_n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    };

    for (int iteration = 0; iteration < options.gemm_warmup; ++iteration) {
        gemm();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto milliseconds = measure_samples(options.gemm_samples, 1, gemm);

    __half first_result;
    __half last_result;
    CUDA_CHECK(cudaMemcpy(
        &first_result, result, sizeof(__half), cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &last_result,
        result + element_count - 1,
        sizeof(__half),
        cudaMemcpyDeviceToHost
    ));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(first));
    CUDA_CHECK(cudaFree(second));
    CUDA_CHECK(cudaFree(result));

    const double operations =
        2.0 * options.gemm_n * options.gemm_n * options.gemm_n;
    std::vector<double> rates;
    rates.reserve(milliseconds.size());
    for (const double elapsed : milliseconds) {
        rates.push_back(operations / (elapsed / 1000.0) / 1.0e12);
    }

    const double expected =
        options.gemm_n * std::pow(1.0 / 128.0, 2.0);
    return {
        std::move(milliseconds),
        std::move(rates),
        __half2float(first_result),
        __half2float(last_result),
        expected,
    };
}

BenchmarkResult run_hbm(const Options& options) {
    const std::size_t array_bytes = options.hbm_mib * 1024ULL * 1024ULL;
    const std::size_t element_count = array_bytes / sizeof(float);
    float* first = nullptr;
    float* second = nullptr;
    float* result = nullptr;
    CUDA_CHECK(cudaMalloc(&first, array_bytes));
    CUDA_CHECK(cudaMalloc(&second, array_bytes));
    CUDA_CHECK(cudaMalloc(&result, array_bytes));

    constexpr int threads = 256;
    constexpr int blocks = 4096;
    fill_float<<<blocks, threads>>>(first, element_count, 1.0F);
    fill_float<<<blocks, threads>>>(second, element_count, 2.0F);
    CUDA_CHECK(cudaGetLastError());

    constexpr float scalar = 3.0F;
    const auto hbm_triad = [&]() {
        triad<<<blocks, threads>>>(
            first, second, result, element_count, scalar
        );
        CUDA_CHECK(cudaGetLastError());
    };
    for (int iteration = 0; iteration < options.hbm_warmup; ++iteration) {
        hbm_triad();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto milliseconds = measure_samples(
        options.hbm_samples,
        options.hbm_kernels_per_sample,
        hbm_triad
    );

    float first_result = 0.0F;
    float last_result = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &first_result, result, sizeof(float), cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &last_result,
        result + element_count - 1,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(first));
    CUDA_CHECK(cudaFree(second));
    CUDA_CHECK(cudaFree(result));

    // The triad reads two arrays and writes one array.
    const double transferred_bytes = 3.0 * array_bytes;
    std::vector<double> rates;
    rates.reserve(milliseconds.size());
    for (const double elapsed : milliseconds) {
        rates.push_back(transferred_bytes / (elapsed / 1000.0) / 1.0e9);
    }

    return {
        std::move(milliseconds),
        std::move(rates),
        first_result,
        last_result,
        7.0,
    };
}

void write_samples(
    const Options& options,
    const std::string& host,
    const cudaDeviceProp& properties,
    int driver_version,
    int runtime_version,
    const BenchmarkResult& gemm,
    const BenchmarkResult& hbm
) {
    std::ofstream output(options.output);
    if (!output) {
        throw std::runtime_error("could not open " + options.output);
    }

    output << "# schema_version=1\n";
    output << "# hostname=" << host << '\n';
    output << "# gpu_name=" << properties.name << '\n';
    output << "# gpu_uuid=" << gpu_uuid(properties.uuid) << '\n';
    output << "# cuda_driver_version=" << driver_version << '\n';
    output << "# cuda_runtime_version=" << runtime_version << '\n';
    output << "# slurm_job_id=" << environment_value("SLURM_JOB_ID") << '\n';
    output << "# slurm_cpus_on_node="
           << environment_value("SLURM_CPUS_ON_NODE") << '\n';
    output << "# cuda_visible_devices="
           << environment_value("CUDA_VISIBLE_DEVICES") << '\n';
    output << "# gemm_n=" << options.gemm_n << '\n';
    output << "# gemm_warmup=" << options.gemm_warmup << '\n';
    output << "# gemm_samples=" << options.gemm_samples << '\n';
    output << "# hbm_array_mib=" << options.hbm_mib << '\n';
    output << "# hbm_warmup=" << options.hbm_warmup << '\n';
    output << "# hbm_samples=" << options.hbm_samples << '\n';
    output << "# hbm_kernels_per_sample="
           << options.hbm_kernels_per_sample << '\n';
    output << "metric,sample,milliseconds,value,unit\n";
    output << std::setprecision(10);
    for (std::size_t sample = 0; sample < gemm.rates.size(); ++sample) {
        output << "gemm," << sample << ',' << gemm.milliseconds[sample] << ','
               << gemm.rates[sample] << ",TFLOP/s\n";
    }
    for (std::size_t sample = 0; sample < hbm.rates.size(); ++sample) {
        output << "hbm," << sample << ',' << hbm.milliseconds[sample] << ','
               << hbm.rates[sample] << ",GB/s\n";
    }
}

void assert_correct(const BenchmarkResult& result, const std::string& name) {
    constexpr double tolerance = 1.0e-3;
    if (std::abs(result.first_value - result.expected_value) > tolerance ||
        std::abs(result.last_value - result.expected_value) > tolerance) {
        std::ostringstream message;
        message << name << " correctness check failed: expected "
                << result.expected_value << ", got " << result.first_value
                << " and " << result.last_value;
        throw std::runtime_error(message.str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        int driver_version = 0;
        int runtime_version = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        const std::string host = hostname();

        std::cout << "PILOT_SCHEMA_VERSION=1\n";
        std::cout << "HOSTNAME=" << host << '\n';
        std::cout << "GPU_NAME=" << properties.name << '\n';
        std::cout << "GPU_UUID=" << gpu_uuid(properties.uuid) << '\n';
        std::cout << "GPU_COMPUTE_CAPABILITY=" << properties.major << '.'
                  << properties.minor << '\n';
        std::cout << "GPU_GLOBAL_MEMORY_BYTES=" << properties.totalGlobalMem
                  << '\n';
        std::cout << "GPU_MEMORY_CLOCK_KHZ=" << properties.memoryClockRate
                  << '\n';
        std::cout << "GPU_MEMORY_BUS_WIDTH_BITS="
                  << properties.memoryBusWidth << '\n';
        std::cout << "CUDA_DRIVER_VERSION=" << driver_version << '\n';
        std::cout << "CUDA_RUNTIME_VERSION=" << runtime_version << '\n';
        std::cout << "SLURM_JOB_ID=" << environment_value("SLURM_JOB_ID")
                  << '\n';
        std::cout << "SLURM_CPUS_ON_NODE="
                  << environment_value("SLURM_CPUS_ON_NODE") << '\n';
        std::cout << "CUDA_VISIBLE_DEVICES="
                  << environment_value("CUDA_VISIBLE_DEVICES") << '\n';
        std::cout << "GEMM_N=" << options.gemm_n << '\n';
        std::cout << "GEMM_SAMPLES=" << options.gemm_samples << '\n';
        std::cout << "HBM_ARRAY_MIB=" << options.hbm_mib << '\n';
        std::cout << "HBM_SAMPLES=" << options.hbm_samples << '\n';
        std::cout << "HBM_KERNELS_PER_SAMPLE="
                  << options.hbm_kernels_per_sample << '\n';

        const BenchmarkResult gemm = run_gemm(options);
        assert_correct(gemm, "GEMM");
        const BenchmarkResult hbm = run_hbm(options);
        assert_correct(hbm, "HBM");
        write_samples(
            options,
            host,
            properties,
            driver_version,
            runtime_version,
            gemm,
            hbm
        );

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "GEMM_MEDIAN_TFLOPS="
                  << percentile(gemm.rates, 0.50) << '\n';
        std::cout << "GEMM_P05_TFLOPS="
                  << percentile(gemm.rates, 0.05) << '\n';
        std::cout << "GEMM_CV_PERCENT="
                  << coefficient_of_variation_percent(gemm.rates) << '\n';
        std::cout << "HBM_MEDIAN_GBPS="
                  << percentile(hbm.rates, 0.50) << '\n';
        std::cout << "HBM_P05_GBPS="
                  << percentile(hbm.rates, 0.05) << '\n';
        std::cout << "HBM_CV_PERCENT="
                  << coefficient_of_variation_percent(hbm.rates) << '\n';
        std::cout << "SANITY_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR=" << error.what() << '\n';
        return 1;
    }
}

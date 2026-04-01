#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/tensor.cuh"
#include "../include/kernels.cuh"

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

void bench_shape(cublasHandle_t handle, int batch, int in_f, int out_f) {
    Tensor<float> A(batch * in_f,  Device::GPU);
    Tensor<float> B(out_f * in_f,  Device::GPU);
    Tensor<float> bias(out_f,      Device::GPU);
    Tensor<float> C(batch * out_f, Device::GPU);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 50;
    float bytes = (float)(batch*in_f + out_f*in_f + batch*out_f) * sizeof(float);

    auto bench = [&](auto fn) {
        for (int i = 0; i < 5; i++) fn();  // warmup
        cudaEventRecord(start);
        for (int i = 0; i < RUNS; i++) fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        return elapsed_ms(start, stop) / RUNS;
    };

    float naive_ms  = bench([&]{ launch_matmul_naive (A.data, B.data, bias.data, C.data, batch, in_f, out_f); });
    float tiled_ms  = bench([&]{ launch_matmul_tiled (A.data, B.data, bias.data, C.data, batch, in_f, out_f); });
    float cublas_ms = bench([&]{ launch_matmul_cublas(handle, A.data, B.data, bias.data, C.data, batch, in_f, out_f); });

    auto bw = [&](float ms) { return (bytes / 1e9f) / (ms / 1000.0f); };

    std::cout << "[" << batch << "x" << in_f << "] x ["
              << out_f << "x" << in_f << "]\n";
    std::cout << "  Naive:   " << std::fixed << std::setprecision(3)
              << naive_ms  << " ms  " << std::setprecision(1) << bw(naive_ms)  << " GB/s\n";
    std::cout << "  Tiled:   " << std::setprecision(3)
              << tiled_ms  << " ms  " << std::setprecision(1) << bw(tiled_ms)  << " GB/s\n";
    std::cout << "  cuBLAS:  " << std::setprecision(3)
              << cublas_ms << " ms  " << std::setprecision(1) << bw(cublas_ms) << " GB/s\n";
    std::cout << "  Speedup vs naive:  " << std::setprecision(2)
              << naive_ms / cublas_ms << "x\n";
    std::cout << "  Speedup vs tiled:  "
              << tiled_ms / cublas_ms << "x\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "=== Matmul Benchmark: Naive vs Tiled vs cuBLAS ===\n\n";

    bench_shape(handle, 1024, 784,  128);
    bench_shape(handle, 1024, 128,  64);
    bench_shape(handle, 4096, 1024, 1024);
    bench_shape(handle, 8192, 2048, 512);

    cublasDestroy(handle);
    return 0;
}
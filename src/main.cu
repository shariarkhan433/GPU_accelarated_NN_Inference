#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/tensor.cuh"
#include "../include/kernels.cuh"

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

float bench_kernel(bool tiled,
                   float* A, float* B, float* bias, float* C,
                   int batch, int in_f, int out_f,
                   int runs=50) {
    // warmup
    for (int i = 0; i < 5; i++) {
        if (tiled) launch_matmul_tiled(A, B, bias, C, batch, in_f, out_f);
        else       launch_matmul_naive(A, B, bias, C, batch, in_f, out_f);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        if (tiled) launch_matmul_tiled(A, B, bias, C, batch, in_f, out_f);
        else       launch_matmul_naive(A, B, bias, C, batch, in_f, out_f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = elapsed_ms(start, stop) / runs;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void bench_shape(int batch, int in_f, int out_f) {
    // Allocate on GPU
    Tensor<float> A(batch * in_f,   Device::GPU);
    Tensor<float> B(out_f * in_f,   Device::GPU);
    Tensor<float> bias(out_f,       Device::GPU);
    Tensor<float> C(batch * out_f,  Device::GPU);

    float naive_ms = bench_kernel(false,
        A.data, B.data, bias.data, C.data, batch, in_f, out_f);
    float tiled_ms = bench_kernel(true,
        A.data, B.data, bias.data, C.data, batch, in_f, out_f);

    float speedup = naive_ms / tiled_ms;

    // Compute effective memory bandwidth (GB/s)
    // Each kernel reads A + B, writes C
    float bytes = (float)(batch*in_f + out_f*in_f + batch*out_f) * sizeof(float);
    float naive_bw = (bytes / 1e9f) / (naive_ms / 1000.0f);
    float tiled_bw = (bytes / 1e9f) / (tiled_ms / 1000.0f);

    std::cout << "[" << batch << "x" << in_f << "] x ["
              << out_f << "x" << in_f << "]\n";
    std::cout << "  Naive:  " << std::fixed << std::setprecision(3)
              << naive_ms << " ms  " << std::setprecision(1)
              << naive_bw << " GB/s\n";
    std::cout << "  Tiled:  " << std::fixed << std::setprecision(3)
              << tiled_ms << " ms  " << std::setprecision(1)
              << tiled_bw << " GB/s\n";
    std::cout << "  Speedup: " << std::setprecision(2)
              << speedup << "x\n\n";
}

int main() {
    std::cout << "=== Matmul Kernel Benchmark ===\n\n";

    // Small — our actual network layers
    bench_shape(1024, 784, 128);   // fc1
    bench_shape(1024, 128, 64);    // fc2
    bench_shape(1024, 64,  10);    // fc3

    // Large — where tiling really shines
    bench_shape(4096, 1024, 1024);
    bench_shape(8192, 2048, 512);

    return 0;
}
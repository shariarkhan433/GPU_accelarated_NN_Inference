#include <iostream>
#include <iomanip>
#include <cmath>
#include "../include/network.cuh"

// Returns milliseconds between two CUDA events
float cuda_time_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    const size_t BATCH = 1024;  // large enough to measure

    Network<float> net;
    net.to_gpu();

    // Build input on GPU
    Tensor<float> input(BATCH * 784, Device::CPU);
    for (size_t i = 0; i < input.size; i++)
        input.data[i] = 0.01f * (i % 100);
    input.to_gpu();

    // Warmup — GPU needs a few runs to reach stable clock speed
    for (int i = 0; i < 3; i++)
        net.forward_gpu(input, BATCH);

    // Time it with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 20;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        net.forward_gpu(input, BATCH);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = cuda_time_ms(start, stop);
    float avg_ms   = total_ms / RUNS;

    std::cout << "Batch size:       " << BATCH << "\n";
    std::cout << "Runs:             " << RUNS << "\n";
    std::cout << "Avg forward pass: " << std::fixed
              << std::setprecision(3) << avg_ms << " ms\n";
    std::cout << "Throughput:       "
              << (int)(BATCH / (avg_ms / 1000.0f))
              << " samples/sec\n";

    // Correctness check on smaller batch
    Network<float> net2;
    Tensor<float> in_cpu(64 * 784, Device::CPU);
    for (size_t i = 0; i < in_cpu.size; i++)
        in_cpu.data[i] = 0.01f * (i % 100);
    Tensor<float> cpu_out = net2.forward_cpu(in_cpu, 64);

    net2.to_gpu();
    Tensor<float> in_gpu(64 * 784, Device::CPU);
    for (size_t i = 0; i < in_gpu.size; i++)
        in_gpu.data[i] = 0.01f * (i % 100);
    in_gpu.to_gpu();
    Tensor<float> gpu_out = net2.forward_gpu(in_gpu, 64);
    gpu_out.to_cpu();

    float max_diff = 0;
    for (size_t i = 0; i < 64 * 10; i++)
        max_diff = std::max(max_diff,
                   std::abs(cpu_out.data[i] - gpu_out.data[i]));
    std::cout << "\nCorrectness max diff: " << max_diff << "\n";
    std::cout << (max_diff < 1e-4f ? "PASS" : "FAIL") << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
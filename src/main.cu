#include <iostream>
#include <iomanip>
#include <cmath>
#include "../include/network.cuh"

int main() {
    Network<float> net;

    // Input: batch of 4, all 0.1f
    Tensor<float> input_cpu(4 * 784, Device::CPU);
    for (size_t i = 0; i < input_cpu.size; i++)
        input_cpu.data[i] = 0.1f;

    // CPU forward pass
    Tensor<float> cpu_out = net.forward_cpu(input_cpu, 4);
    std::cout << "CPU output (sample 0):\n";
    for (size_t i = 0; i < 10; i++)
        std::cout << std::fixed << std::setprecision(4)
                  << cpu_out.data[i] << " ";
    std::cout << "\n";

    // Move network and input to GPU
    net.to_gpu();
    Tensor<float> input_gpu(4 * 784, Device::GPU);
    // Copy input to GPU
    Tensor<float> input_cpu2(4 * 784, Device::CPU);
    for (size_t i = 0; i < input_cpu2.size; i++)
        input_cpu2.data[i] = 0.1f;
    input_cpu2.to_gpu();

    // GPU forward pass
    Tensor<float> gpu_out = net.forward_gpu(input_cpu2, 4);

    // Bring GPU output back to CPU to compare
    gpu_out.to_cpu();
    std::cout << "GPU output (sample 0):\n";
    for (size_t i = 0; i < 10; i++)
        std::cout << std::fixed << std::setprecision(4)
                  << gpu_out.data[i] << " ";
    std::cout << "\n";

    // Check max difference between CPU and GPU outputs
    float max_diff = 0;
    for (size_t i = 0; i < 10; i++)
        max_diff = std::max(max_diff,
                   std::abs(cpu_out.data[i] - gpu_out.data[i]));
    std::cout << "\nMax difference CPU vs GPU: " << max_diff << "\n";
    std::cout << (max_diff < 1e-4 ? "PASS" : "FAIL") << "\n";

    return 0;
}
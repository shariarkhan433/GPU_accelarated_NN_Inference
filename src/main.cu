#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/network.cuh"
#include "../include/npy.cuh"

int main() {
    // --- Load test data ---
    std::vector<size_t> shape;
    Tensor<float> images = load_npy("weights/test_images.npy", shape);
    std::cout << "Test images: [" << shape[0] << ", " << shape[1] << "]\n";

    Tensor<float> labels_f = load_npy("weights/test_labels.npy", shape);
    std::cout << "Test labels: [" << shape[0] << "]\n";

    size_t n_samples = 1000;

    // Convert float labels to int
    std::vector<int> labels(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        labels[i] = (int)labels_f.data[i];

    // --- Build network and load weights ---
    Network<float> net;
    net.load_weights("weights");
    
// Debug: print first 5 weights of fc1
std::cout << "fc1 w[0..4]: ";
for (int i = 0; i < 5; i++)
    std::cout << net.fc1.weights.data[i] << " ";
std::cout << "\n";

// Debug: print first 5 values of test image 0
std::cout << "image[0][0..4]: ";
for (int i = 0; i < 5; i++)
    std::cout << images.data[i] << " ";
std::cout << "\n";
    // --- CPU inference ---
    Tensor<float> cpu_out = net.forward_cpu(images, n_samples);

    int cpu_correct = 0;
    for (size_t i = 0; i < n_samples; i++) {
        // Find argmax of output row i
        int pred = 0;
        float best = cpu_out.data[i * 10];
        for (int j = 1; j < 10; j++) {
            if (cpu_out.data[i * 10 + j] > best) {
                best = cpu_out.data[i * 10 + j];
                pred = j;
            }
        }
        if (pred == labels[i]) cpu_correct++;
    }
    std::cout << "\nCPU inference:\n";
    std::cout << "  Correct: " << cpu_correct << "/" << n_samples << "\n";
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
              << (cpu_correct * 100.0f / n_samples) << "%\n";

    // --- GPU inference ---
    net.to_gpu();
    images.to_gpu();
    Tensor<float> gpu_out = net.forward_gpu(images, n_samples);
    gpu_out.to_cpu();

    int gpu_correct = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int pred = 0;
        float best = gpu_out.data[i * 10];
        for (int j = 1; j < 10; j++) {
            if (gpu_out.data[i * 10 + j] > best) {
                best = gpu_out.data[i * 10 + j];
                pred = j;
            }
        }
        if (pred == labels[i]) gpu_correct++;
    }
    std::cout << "\nGPU inference:\n";
    std::cout << "  Correct: " << gpu_correct << "/" << n_samples << "\n";
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
              << (gpu_correct * 100.0f / n_samples) << "%\n";

    // --- Summary ---
    std::cout << "\n=== Summary ===\n";
    std::cout << "PyTorch baseline: 97.34%\n";
    std::cout << "CPU inference:    " << std::fixed << std::setprecision(2)
              << (cpu_correct * 100.0f / n_samples) << "%\n";
    std::cout << "GPU inference:    "
              << (gpu_correct * 100.0f / n_samples) << "%\n";
    std::cout << "CPU==GPU: "
              << (cpu_correct == gpu_correct ? "yes" : "no") << "\n";

    return 0;
}
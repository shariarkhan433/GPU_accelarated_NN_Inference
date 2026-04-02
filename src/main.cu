
#include <iostream>
#include <iomanip>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/network.cuh"
#include "../include/npy.cuh"

int main() {
    
    std::vector<size_t> shape;
    Tensor<float> images = load_npy("weights/test_images.npy", shape);
    std::cout << "Test images: [" << shape[0] << ", " << shape[1] << "]\n";

    
    size_t n_samples = 1000;
    std::vector<int> labels(n_samples);
    {
        std::ifstream f("weights/test_labels.npy", std::ios::binary);
        // npy header: 6-byte magic + 2-byte version + 2-byte header_len + header_len bytes
        uint8_t magic[6]; f.read((char*)magic, 6);
        uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
        uint16_t hlen; f.read((char*)&hlen, 2);
        std::string header(hlen, ' '); f.read(header.data(), hlen);
        // raw data: n_samples × int64 little-endian
        for (size_t i = 0; i < n_samples; i++) {
            int64_t v; f.read((char*)&v, 8);
            labels[i] = (int)v;
        }
        std::cout << "Test labels: [" << n_samples << "]  (first 5: ";
        for (int i = 0; i < 5; i++) std::cout << labels[i] << " ";
        std::cout << ")\n";
    }

    
    Network<float> net;
    net.load_weights("weights");

    // --- CPU inference ---
    Tensor<float> cpu_out = net.forward_cpu(images, n_samples);

    int cpu_correct = 0;
    for (size_t i = 0; i < n_samples; i++) {
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

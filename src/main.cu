#include <iostream>
#include <iomanip>
#include "../include/network.cuh"

int main() {
    Network<float> net;
    std::cout << "Network created\n";
    std::cout << "fc1 weights device: "
              << (net.fc1.weights.device == Device::CPU ? "CPU" : "GPU") << "\n";

    // Fake input: batch of 2 MNIST-shaped images [2, 784]
    // All zeros — we just want to verify the forward pass runs
    Tensor<float> input(2 * 784, Device::CPU);
    input.fill(0.0f);

    // Run forward pass
    Tensor<float> output = net.forward_cpu(input, 2);

    // Output should be [2, 10] probabilities
    std::cout << "\nOutput probabilities [2 samples, 10 classes]:\n";
    for (size_t b = 0; b < 2; b++) {
        std::cout << "Sample " << b << ": ";
        float sum = 0;
        for (size_t i = 0; i < 10; i++) {
            float val = output.data[b * 10 + i];
            std::cout << std::fixed << std::setprecision(3) << val << " ";
            sum += val;
        }
        std::cout << "| sum=" << std::fixed << std::setprecision(4) << sum << "\n";
    }

    // Move to GPU
    net.to_gpu();
    std::cout << "\nfc1 weights device after to_gpu(): "
              << (net.fc1.weights.device == Device::CPU ? "CPU" : "GPU") << "\n";

    return 0;
}
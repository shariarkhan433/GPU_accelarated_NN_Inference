#include <iostream>
#include "../include/tensor.cuh"
#include "../include/linear.cuh"

using namespace std;

int main() {
    // --- Tensor test (keep from before) ---
    Tensor<float> t(6, Device::CPU);
    t.fill(3.14f);
    cout << "CPU tensor: ";
    for (size_t i = 0; i < t.size; i++)
        cout << t.data[i] << " ";
    cout << "\n";

    t.to_gpu();
    t.to_cpu();
    cout << "Round-trip GPU->CPU: ";
    for (size_t i = 0; i < t.size; i++)
        cout << t.data[i] << " ";
    cout << "\n";

    // --- Linear layer test ---
    // 3 inputs, 2 outputs — tiny layer to inspect manually
    Linear<float> layer(3, 2);

    cout << "\nWeights [2x3] after xavier init:\n";
    for (size_t i = 0; i < layer.out_features; i++) {
        for (size_t j = 0; j < layer.in_features; j++)
            cout << layer.weights.data[i * layer.in_features + j] << "\t";
        cout << "\n";
    }

    cout << "\nBias [2] after init:\n";
    for (size_t i = 0; i < layer.out_features; i++)
        cout << layer.bias.data[i] << " ";
    cout << "\n";

    // Move to GPU
    layer.to_gpu();
    cout << "\nLayer moved to GPU\n";
    cout << "weights.device is GPU: "
              << (layer.weights.device == Device::GPU ? "yes" : "no") << "\n";
    cout << "bias.device is GPU: "
              << (layer.bias.device == Device::GPU ? "yes" : "no") << "\n";

    return 0;
}
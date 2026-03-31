#include <iostream>
#include "../include/tensor.cuh"
#include "../include/linear.cuh"
#include "../include/activations.cuh"

void print_tensor(const char* label, Tensor<float>& t) {
    std::cout << label << ": ";
    for (size_t i = 0; i < t.size; i++)
        std::cout << t.data[i] << " ";
    std::cout << "\n";
}

int main() {
    // --- ReLU test ---
    Tensor<float> x(6, Device::CPU);
    // manually set mixed positive/negative values
    float vals[] = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f};
    for (size_t i = 0; i < 6; i++) x.data[i] = vals[i];

    print_tensor("Before ReLU", x);
    ReLU<float> relu;
    relu.forward_cpu(x);
    print_tensor("After ReLU ", x);

    // --- Softmax test ---
    // 2 samples, 3 classes each → batch_size=2, out_features=3
    Tensor<float> logits(6, Device::CPU);
    float raw[] = {1.0f, 2.0f, 3.0f,   // sample 1 — class 2 should dominate
                   0.5f, 0.5f, 0.1f};  // sample 2 — classes 0 and 1 tied
    for (size_t i = 0; i < 6; i++) logits.data[i] = raw[i];

    print_tensor("Before Softmax", logits);
    Softmax<float> softmax(3);
    softmax.forward_cpu(logits, 2);
    print_tensor("After Softmax ", logits);

    // Verify: each row must sum to 1.0
    float sum0 = logits.data[0] + logits.data[1] + logits.data[2];
    float sum1 = logits.data[3] + logits.data[4] + logits.data[5];
    std::cout << "Row 0 sum: " << sum0 << " (must be 1.0)\n";
    std::cout << "Row 1 sum: " << sum1 << " (must be 1.0)\n";

    // --- Linear layer still works ---
    Linear<float> layer(3, 2);
    layer.to_gpu();
    std::cout << "\nLinear layer on GPU: "
              << (layer.weights.device == Device::GPU ? "yes" : "no") << "\n";

    return 0;
}
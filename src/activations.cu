#include "../include/activations.cuh"
#include <algorithm>
#include <cmath>

template <typename T>
void ReLU<T>::forward_cpu(Tensor<T>& x) {
    if (x.device != Device::CPU)
        throw std::runtime_error("ReLU::forward_cpu called on GPU tensor");

    for (size_t i = 0; i < x.size; i++)
        x.data[i] = std::max((T)0, x.data[i]);
}

template <typename T>
void Softmax<T>::forward_cpu(Tensor<T>& x, size_t batch_size) {
    if (x.device != Device::CPU)
        throw std::runtime_error("Softmax::forward_cpu called on GPU tensor");

    // Process one row (one sample) at a time
    for (size_t b = 0; b < batch_size; b++) {
        T* row = x.data + b * out_features;

        // Step 1: subtract max for numerical stability
        // Without this, exp() can overflow to inf for large values
        T max_val = *std::max_element(row, row + out_features);
        for (size_t i = 0; i < out_features; i++)
            row[i] = std::exp(row[i] - max_val);

        // Step 2: divide by sum
        T sum = 0;
        for (size_t i = 0; i < out_features; i++)
            sum += row[i];
        for (size_t i = 0; i < out_features; i++)
            row[i] /= sum;
    }
}

template class ReLU<float>;
template class Softmax<float>;
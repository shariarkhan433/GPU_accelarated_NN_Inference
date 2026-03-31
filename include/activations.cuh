#pragma once
#include "tensor.cuh"
#include <cmath>

template <typename T>
class ReLU {
    public:
    //ReLU f(x) = max(0,x)
    // Operates in_place - modifies tensor directly, no new allcation
    void forward_cpu(Tensor<T>& x);
};

template <typename T>
class Softmax {
    public:
    // Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
    // Applied per-row for batched input
    // out_features = number of classes
    size_t out_features;

    Softmax(size_t out_features) : out_features(out_features) {}

    void forward_cpu(Tensor<T>& x, size_t batch_size);
};
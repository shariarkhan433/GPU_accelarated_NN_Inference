#pragma once
#include "linear.cuh"
#include "activations.cuh"
#include <vector>

template <typename T>
class Network {
    public:
        //layer -- stored by value, move-contructor in
        Linear<T> fc1;
        Linear<T> fc2;
        Linear<T> fc3;

        ReLU<T> relu;
        Softmax<T> softmax;

        Network();

        //move all layers to GPU
        void to_gpu();

        //full forward pass on CPU
        Tensor<T> forward_cpu(Tensor<T>& input, size_t batch_size);
};
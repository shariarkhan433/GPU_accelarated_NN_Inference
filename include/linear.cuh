#pragma once
#include "tensor.cuh"

template <typename T>
class Linear {
    public:
        size_t in_features;
        size_t out_features;

        Tensor<T> weights; //share: [out_features, in_features]
        Tensor<T> bias;     //shapre: [out_features]

        //constructor -- allocates weights and bias on CPU then move to GPU
        Linear(size_t in_features, size_t out_features);

        //Tensor already deltes the copy at this point
        Linear(const Linear&)=delete;
        Linear& operator=(const Linear&) = delete;

        //move
        Linear(Linear&&) = default;
        Linear& operator=(Linear&&)=default;

        //Initializing weights with xavier uniform
        void xavier_init();

        //Move both weights and bias to GPU
        void to_gpu();
};
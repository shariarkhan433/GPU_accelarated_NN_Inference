#pragma once
#include <cuda_runtime.h>

//Naive matmul

void launch_matmul_naive(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f
);

void launch_matmul_tiled(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f);

// ReLU in-place: x[i] = max(0, x[i])
void launch_relu(float* x, int n);

// Softmax in-place: per row, batch_size rows of width n_classes
void launch_softmax(float* x, int batch_size, int n_classes);
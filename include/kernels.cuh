#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cuBLAS handle — create once, reuse
// Declared extern so main.cu can pass it in
void launch_matmul_cublas(
    cublasHandle_t handle,
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f);

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

__global__ void add_bias(float* C, float* bias, int batch, int out_f);
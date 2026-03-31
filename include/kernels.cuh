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

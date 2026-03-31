#include "../include/kernels.cuh"
// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
// #include <stdio.h>

#define TILE_SIZE 16

// Naive kernel
// Each thread computes one element of C

__global__ void matmul_naive_kernel(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //guarding so doesn't go out of bound
    if(row>=batch || col>=out_f) return;
    
    //Dot product
    float sum=bias[col];
    for (int k=0; k<in_f; k++) {
        sum+= A[row*in_f +k] * B[col*in_f+k];
    }
    
    C[row*out_f +col]=sum;
}

void launch_matmul_naive(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f
)
{
    //Each block has 16*16 threads
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    //
    dim3 grid(
        (out_f+TILE_SIZE -1) / TILE_SIZE,
        (batch+TILE_SIZE -1) / TILE_SIZE
    );
    matmul_naive_kernel<<<grid, block>>>(A, B, bias, C, batch, in_f, out_f);
    cudaDeviceSynchronize();
}


__global__ void matmul_tiled_kernel(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f
){
    //step 2
}


void launch_matmul_tiled(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f
)
{
    //step 2
}


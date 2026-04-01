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
    int batch, int in_f, int out_f)
{
    // Shared memory tiles — live in fast on-chip memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // batch index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // out_f index

    float sum = 0;

    // Number of tiles needed to cover in_f dimension
    int n_tiles = (in_f + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; t++) {
        // Each thread loads one element into shared memory
        // Tile covers columns [t*TILE_SIZE .. (t+1)*TILE_SIZE-1] of A and B

        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_col = t * TILE_SIZE + threadIdx.y;

        // Bounds check before loading
        tileA[threadIdx.y][threadIdx.x] =
            (row < batch && a_col < in_f) ? A[row * in_f + a_col] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (col < out_f && b_col < in_f) ? B[col * in_f + b_col] : 0.0f;

        // Wait until ALL threads in block have loaded their element
        __syncthreads();

        // Compute partial dot product from this tile
        // All reads are from shared memory — fast
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Wait before loading next tile — don't overwrite while others still read
        __syncthreads();
    }

    // Write result
    if (row < batch && col < out_f)
        C[row * out_f + col] = sum + bias[col];
}

void launch_matmul_tiled(
    float* A, float* B, float* bias, float* C,
    int batch, int in_f, int out_f)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_f + TILE_SIZE - 1) / TILE_SIZE,
        (batch + TILE_SIZE - 1) / TILE_SIZE
    );
    matmul_tiled_kernel<<<grid, block>>>(A, B, bias, C, batch, in_f, out_f);
    cudaDeviceSynchronize();
}


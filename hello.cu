#include<iostream>

__global__ void hello(){
    printf("GPU thread %d available\n", threadIdx.x);
}

int main(){
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
#include "../include/tensor.cuh"
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

template <typename T>
Tensor<T>:: Tensor(size_t size, Device device)
    : size(size), device(device), data(nullptr)
    {
        if (device == Device::CPU) {
        data = new T[size];
        }else {
        cudaError_t err= cudaMalloc(&data, size * sizeof(T));
        if(err != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    //RAII in action -- no manual cleanup ever needed by the caller
    template <typename T>
    Tensor<T>::~Tensor() {
        if(data ==nullptr) return;
        if(device == Device::CPU)
            delete[] data;
        else
            cudaFree(data);
    }
// Move constructor — steal the other tensor's data, leave it empty
template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : size(other.size), device(other.device), data(other.data)
{
    other.data = nullptr;  // critical: prevent double-free
    other.size = 0;
}

    //move assignment
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if(this == &other) return *this;
    //freeing our current data
    if(data){
        if(device ==Device::CPU) delete[] data;
        else{
            cudaFree(data);
        }
    }
    data = other.data;
    size = other.size;
    device = other.device;
    other.data = nullptr;
    other.size = 0;
    return *this;
    }

//in the CPU for now
template <typename T>
void Tensor<T>::fill(T value) {
    if (device != Device::CPU)
        throw std::runtime_error("fill() only on CPU tensor");
    for (size_t i = 0; i < size; i++)
        data[i] = value;
}

//GPU transfer
template <typename T>
void Tensor<T>::to_gpu() {
    if (device == Device::GPU) return;
    T* gpu_data;
    cudaMalloc(&gpu_data, size * sizeof(T));
    cudaMemcpy(gpu_data, data, size * sizeof(T), cudaMemcpyHostToDevice);
    delete[] data;
    data = gpu_data;
    device = Device::GPU;
}

//CPU transfer
template <typename T>
void Tensor<T>::to_cpu() {
    if (device == Device::CPU) return;
    T* cpu_data = new T[size];
    cudaMemcpy(cpu_data, data, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(data);
    data = cpu_data;
    device = Device::CPU;
}

//Explicit instruction -- needed for implenmentation in .cu file
template class Tensor<float>;
template class Tensor<int>;
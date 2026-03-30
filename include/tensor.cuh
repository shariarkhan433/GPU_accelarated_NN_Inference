#pragma once
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <utility>

enum class Device {CPU, GPU};

template <typename T>
class Tensor {
    public:
    size_t size;
    Device device;
    T* data;    //raw pointer, we manage this manually

    //constructuor, remember?
    Tensor (size_t size, Device device);

    ~Tensor(); //Destructor -- RAII: Free up memory automatically

    //Delete copies. We dont want accidental copies
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    //Allow move -- Transfer ownership without copying
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Utilities
    void fill(T value); // fill all the elements with a value
    void to_gpu();
    void to_cpu();
};


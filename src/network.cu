#include "../include/network.cuh"
#include <stdexcept>
#include "../include/kernels.cuh" // <-- newly added

template <typename T>
Network<T>::Network()
    : fc1(784, 128),
      fc2(128, 64),
      fc3(64, 10),
      relu(),
      softmax(10)
{}

template<typename T>
void Network<T>::to_gpu(){
    fc1.to_gpu();
    fc2.to_gpu();
    fc3.to_gpu();
}

// CPU matrix multiplication:: out [b, j]= sum_k(in[b,k]*w[j,k]+bias[j])
//w is stored row-major as [out_features, in_features]

static void matmul_cpu(
        float* in,     // [batch, in_f]
    float* W,      // [out_f, in_f]
    float* bias,   // [out_f]
    float* out,    // [batch, out_f]
    size_t batch,
    size_t in_f,
    size_t out_f)
{
for (int b=0;b<batch;b++){
    for (int j=0;j<out_f;j++) {
        float sum=bias[j];
        for (int k=0; k<in_f; k++) {
            sum +=in[b*in_f + k]* W[j*in_f+k];
        }
        out[b*out_f +j]=sum;
    }
}
}

template <typename T>
Tensor<T> Network<T>::forward_gpu(Tensor<T>& input, size_t batch_size) {
    if (input.device != Device::GPU)
        throw std::runtime_error("forward_gpu requires GPU tensor");

    // Layer 1: [batch, 784] -> [batch, 128]
    Tensor<T> out1(batch_size * 128, Device::GPU);
    launch_matmul_naive(input.data, fc1.weights.data, fc1.bias.data,
                        out1.data, (int)batch_size, 784, 128);
    out1.to_cpu(); relu.forward_cpu(out1); out1.to_gpu();

    // Layer 2: [batch, 128] -> [batch, 64]
    Tensor<T> out2(batch_size * 64, Device::GPU);
    launch_matmul_naive(out1.data, fc2.weights.data, fc2.bias.data,
                        out2.data, (int)batch_size, 128, 64);
    out2.to_cpu(); relu.forward_cpu(out2); out2.to_gpu();

    // Layer 3: [batch, 64] -> [batch, 10]
    Tensor<T> out3(batch_size * 10, Device::GPU);
    launch_matmul_naive(out2.data, fc3.weights.data, fc3.bias.data,
                        out3.data, (int)batch_size, 64, 10);
    out3.to_cpu(); softmax.forward_cpu(out3, batch_size); out3.to_gpu();

    return out3;
}
template <typename T>
Tensor<T> Network<T>::forward_cpu(Tensor<T>& input, size_t batch_size) {
    if (input.device != Device::CPU)
        throw std::runtime_error("forward_cpu requires CPU tensor");

    // Layer 1: [batch, 784] -> [batch, 128]
    Tensor<T> out1(batch_size * 128, Device::CPU);
    matmul_cpu(input.data, fc1.weights.data, fc1.bias.data,
               out1.data, batch_size, 784, 128);
    relu.forward_cpu(out1);

    // Layer 2: [batch, 128] -> [batch, 64]
    Tensor<T> out2(batch_size * 64, Device::CPU);
    matmul_cpu(out1.data, fc2.weights.data, fc2.bias.data,
               out2.data, batch_size, 128, 64);
    relu.forward_cpu(out2);

    // Layer 3: [batch, 64] -> [batch, 10]
    Tensor<T> out3(batch_size * 10, Device::CPU);
    matmul_cpu(out2.data, fc3.weights.data, fc3.bias.data,
               out3.data, batch_size, 64, 10);
    softmax.forward_cpu(out3, batch_size);

    return out3;  // move semantics kicks in here — no copy
}

template class Network<float>;
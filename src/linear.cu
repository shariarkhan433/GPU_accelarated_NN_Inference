#include "../include/linear.cuh"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>

using namespace std;

template <typename T>
Linear<T>::Linear(size_t in_features, size_t out_features)
        : in_features(in_features), out_features(out_features),
        weights(in_features * out_features, Device::CPU),
        bias(out_features, Device::CPU)
        {
            xavier_init();
        }

template<typename T>
void Linear<T>::xavier_init() {
    //xavier uniform: values drawn form [-limit, limit]
    //limit = sqrt(6 / (in+out)) - keeps gradients stable at init
    T limit = sqrt(6.0f/(in_features+out_features));

    static bool seeded = false;
    if (!seeded) {
    srand(42);
    seeded=true;
    }
    for (size_t i=0; i<in_features*out_features; i++) {
    T rand_val = (T)::rand() / RAND_MAX; // [0,1]
    weights.data[i] = rand_val*2*limit - limit;
    }
    // Bias initialized to zero — standard practice
    for (size_t i = 0; i < out_features; i++)
        bias.data[i] = 0;
}

template<typename T>
void Linear<T>::to_gpu(){
    weights.to_gpu();
    bias.to_gpu();
}
template class Linear<float>;
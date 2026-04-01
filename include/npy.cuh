#pragma once
#include "tensor.cuh"
#include <string>
#include <vector>

// Loads a .npy file into a CPU Tensor<float>
// Handles both float32 and any shape
Tensor<float> load_npy(const std::string& path, std::vector<size_t>& shape);
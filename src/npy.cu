#include "../include/npy.cuh"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <cstdint>

Tensor<float> load_npy(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);

// --- Magic + version (10 bytes) ---
char magic[7] = {0};
f.read(magic, 6);
if ((unsigned char)magic[0] != 0x93 ||
    magic[1] != 'N' || magic[2] != 'U' ||
    magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
    throw std::runtime_error("Not a .npy file: " + path);

    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);

    // --- Header length ---
    uint32_t header_len;
    if (major == 1) {
        uint16_t hlen;
        f.read((char*)&hlen, 2);
        header_len = hlen;
    } else {
        f.read((char*)&header_len, 4);
    }

    // --- Parse header string ---
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    // Extract shape — find "shape': (" and parse numbers until ')'
    shape.clear();
    size_t pos = header.find("'shape'");
    if (pos == std::string::npos)
        pos = header.find("\"shape\"");
    pos = header.find('(', pos);
    size_t end = header.find(')', pos);
    std::string shape_str = header.substr(pos + 1, end - pos - 1);

    // Parse comma-separated integers
    std::string token;
    for (char c : shape_str) {
        if (c == ',') {
            if (!token.empty()) {
                shape.push_back(std::stoul(token));
                token.clear();
            }
        } else if (c != ' ') {
            token += c;
        }
    }
    if (!token.empty())
        shape.push_back(std::stoul(token));

    // Total elements
    size_t total = std::accumulate(
        shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());

    // --- Read raw float32 data ---
    Tensor<float> t(total, Device::CPU);
    f.read((char*)t.data, total * sizeof(float));

    if (!f)
        throw std::runtime_error("Failed to read data from: " + path);

    return t;
}
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MnistData {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::uint8_t> images; // Images stored consecutively
    std::vector<std::uint8_t> labels;
};

MnistData load_mnist(const std::string& image_path,
                     const std::string& label_path);
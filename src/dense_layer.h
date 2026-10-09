#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

enum class Activation {
    None,
    Relu
};

struct LayerGradients {
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> input_gradient;
};

class DenseLayer {
public:
    DenseLayer(std::size_t input_size,
               std::size_t output_size,
               Activation activation,
               unsigned int seed);

    std::vector<float> forward(const std::vector<float>& input) const;

    LayerGradients backward(
        const std::vector<float>& input,
        const std::vector<float>& output_gradient) const;

    void apply_gradients(const LayerGradients& gradients,
                         float learning_rate);

    std::uint64_t version() const;

private:
    std::size_t input_size_;
    std::size_t output_size_;
    Activation activation_;

    // Each output neuron's weights are stored contiguously.
    std::vector<float> weights_;
    std::vector<float> biases_;
    std::uint64_t version_ = 0;
};
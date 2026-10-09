#include "dense_layer.h"

#include <cmath>
#include <random>
#include <stdexcept>

DenseLayer::DenseLayer(std::size_t input_size,
                       std::size_t output_size,
                       Activation activation,
                       unsigned int seed)
    : input_size_(input_size),
      output_size_(output_size),
      activation_(activation),
      weights_(input_size * output_size),
      biases_(output_size, 0.0f) {
    if (input_size_ == 0 || output_size_ == 0) {
        throw std::invalid_argument("Dense layer dimensions must be positive");
    }

    const float scale = std::sqrt(
        (activation_ == Activation::Relu ? 2.0f : 1.0f) /
        static_cast<float>(input_size_));

    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, scale);

    for (float& weight : weights_) {
        weight = distribution(generator);
    }
}

std::vector<float> DenseLayer::forward(
    const std::vector<float>& input) const {
    if (input.size() != input_size_) {
        throw std::invalid_argument("Dense layer received an input of wrong size");
    }

    std::vector<float> output(output_size_);

    for (std::size_t neuron = 0; neuron < output_size_; ++neuron) {
        float value = biases_[neuron];

        for (std::size_t i = 0; i < input_size_; ++i) {
            value += weights_[neuron * input_size_ + i] * input[i];
        }

        if (activation_ == Activation::Relu && value < 0.0f) {
            value = 0.0f;
        }

        output[neuron] = value;
    }

    return output;
}

LayerGradients DenseLayer::backward(
    const std::vector<float>& input,
    const std::vector<float>& output_gradient) const {
    if (input.size() != input_size_) {
        throw std::invalid_argument("Dense layer backward input has wrong size");
    }
    if (output_gradient.size() != output_size_) {
        throw std::invalid_argument(
            "Dense layer output gradient has wrong size");
    }

    LayerGradients gradients;
    gradients.weights.resize(weights_.size());
    gradients.biases.resize(biases_.size());
    gradients.input_gradient.assign(input_size_, 0.0f);

    for (std::size_t neuron = 0; neuron < output_size_; ++neuron) {
        float preactivation = biases_[neuron];

        for (std::size_t i = 0; i < input_size_; ++i) {
            preactivation += weights_[neuron * input_size_ + i] * input[i];
        }

        float delta = output_gradient[neuron];
        if (activation_ == Activation::Relu && preactivation <= 0.0f) {
            delta = 0.0f;
        }

        gradients.biases[neuron] = delta;

        for (std::size_t i = 0; i < input_size_; ++i) {
            const std::size_t weight_index = neuron * input_size_ + i;
            gradients.weights[weight_index] = delta * input[i];
            gradients.input_gradient[i] += weights_[weight_index] * delta;
        }
    }

    return gradients;
}

void DenseLayer::apply_gradients(const LayerGradients& gradients,
                                float learning_rate) {
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "Learning rate must be positive and finite");
    }
    if (gradients.weights.size() != weights_.size() ||
        gradients.biases.size() != biases_.size()) {
        throw std::invalid_argument("Gradient sizes do not match dense layer");
    }

    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * gradients.weights[i];
    }

    for (std::size_t i = 0; i < biases_.size(); ++i) {
        biases_[i] -= learning_rate * gradients.biases[i];
    }

    ++version_;
}

std::uint64_t DenseLayer::version() const {
    return version_;
}
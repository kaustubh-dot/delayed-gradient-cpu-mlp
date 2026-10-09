#include "dense_layer.h"
#include "mnist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Evaluation {
    double average_loss = 0.0;
    double accuracy = 0.0;
};

struct LossAndGradient {
    float loss = 0.0f;
    std::vector<float> output_gradient;
};

std::vector<float> make_input(const MnistData& data, std::size_t image_index) {
    const std::size_t image_size =
        static_cast<std::size_t>(data.rows) * data.cols;
    const std::size_t start = image_index * image_size;

    std::vector<float> input(image_size);
    for (std::size_t i = 0; i < image_size; ++i) {
        input[i] = static_cast<float>(data.images[start + i]) / 255.0f;
    }

    return input;
}

LossAndGradient calculate_loss_and_gradient(
    const std::vector<float>& logits,
    std::size_t correct_label) {
    if (logits.empty() || correct_label >= logits.size()) {
        throw std::runtime_error("Invalid logits or label");
    }

    const float max_logit =
        *std::max_element(logits.begin(), logits.end());

    double sum_exp = 0.0;
    for (float logit : logits) {
        sum_exp += std::exp(static_cast<double>(logit - max_logit));
    }

    LossAndGradient result;
    result.loss = max_logit
        + static_cast<float>(std::log(sum_exp))
        - logits[correct_label];

    result.output_gradient.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        result.output_gradient[i] = static_cast<float>(
            std::exp(static_cast<double>(logits[i] - max_logit)) / sum_exp);
    }

    result.output_gradient[correct_label] -= 1.0f;
    return result;
}

float train_one(DenseLayer& hidden_layer,
                DenseLayer& output_layer,
                const std::vector<float>& input,
                std::size_t label,
                float learning_rate) {
    const std::vector<float> hidden = hidden_layer.forward(input);
    const std::vector<float> logits = output_layer.forward(hidden);
    const LossAndGradient loss =
        calculate_loss_and_gradient(logits, label);

    // Calculate both gradients before updating either layer.
    const LayerGradients output_gradients =
        output_layer.backward(hidden, loss.output_gradient);
    const LayerGradients hidden_gradients =
        hidden_layer.backward(input, output_gradients.input_gradient);

    output_layer.apply_gradients(output_gradients, learning_rate);
    hidden_layer.apply_gradients(hidden_gradients, learning_rate);

    return loss.loss;
}

Evaluation evaluate(const DenseLayer& hidden_layer,
                    const DenseLayer& output_layer,
                    const MnistData& data,
                    const std::vector<std::size_t>& indices) {
    Evaluation result;
    std::size_t correct = 0;

    for (std::size_t image_index : indices) {
        const std::vector<float> input = make_input(data, image_index);
        const std::vector<float> hidden = hidden_layer.forward(input);
        const std::vector<float> logits = output_layer.forward(hidden);
        const LossAndGradient prediction =
            calculate_loss_and_gradient(logits, data.labels[image_index]);

        result.average_loss += prediction.loss;

        const std::size_t predicted_label =
            static_cast<std::size_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());

        if (predicted_label == data.labels[image_index]) {
            ++correct;
        }
    }

    result.average_loss /= static_cast<double>(indices.size());
    result.accuracy =
        static_cast<double>(correct) / static_cast<double>(indices.size());

    return result;
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 2) {
            std::cerr << "Usage: ./sequential_baseline.exe [epochs]\n";
            return 2;
        }

        std::size_t epochs = 1;
        if (argc == 2) {
            const std::string epoch_argument = argv[1];
            std::size_t parsed_characters = 0;
            epochs = static_cast<std::size_t>(
                std::stoull(epoch_argument, &parsed_characters));

            if (parsed_characters != epoch_argument.size() || epochs < 1 || epochs > 20) {
                throw std::invalid_argument(
                    "Epoch count must be between 1 and 20");
            }
        }

        const MnistData train = load_mnist(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte");

        const MnistData test = load_mnist(
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte");

        constexpr std::size_t training_count = 5000;
        constexpr std::size_t evaluation_count = 1000;
        constexpr float learning_rate = 0.01f;
        constexpr unsigned int seed = 12345;
        constexpr double target_accuracy = 0.90;

        if (train.labels.size() < training_count ||
            test.labels.size() < evaluation_count) {
            throw std::runtime_error("Dataset is smaller than requested subset");
        }

        std::mt19937 generator(seed);

        std::vector<std::size_t> train_indices(train.labels.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), generator);
        train_indices.resize(training_count);

        std::vector<std::size_t> test_indices(test.labels.size());
        std::iota(test_indices.begin(), test_indices.end(), 0);
        std::shuffle(test_indices.begin(), test_indices.end(), generator);
        test_indices.resize(evaluation_count);

        DenseLayer hidden_layer(784, 32, Activation::Relu, seed);
        DenseLayer output_layer(32, 10, Activation::None, seed + 1);

        const Evaluation before =
            evaluate(hidden_layer, output_layer, test, test_indices);

        double total_training_loss = 0.0;
        double training_elapsed_seconds = 0.0;
        double time_to_target_seconds = -1.0;
        Evaluation after = before;

        std::cout << "epoch,average_training_loss,test_loss,"
                     "test_accuracy_percent,elapsed_training_seconds\n";

        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(train_indices.begin(), train_indices.end(), generator);

            double epoch_loss_sum = 0.0;
            const auto epoch_start = std::chrono::steady_clock::now();

            for (std::size_t image_index : train_indices) {
                const std::vector<float> input =
                    make_input(train, image_index);

                epoch_loss_sum += train_one(
                    hidden_layer,
                    output_layer,
                    input,
                    train.labels[image_index],
                    learning_rate);
            }

            const auto epoch_end = std::chrono::steady_clock::now();
            training_elapsed_seconds +=
                std::chrono::duration<double>(epoch_end - epoch_start).count();

            // Evaluation time is excluded from accumulated training time.
            after = evaluate(
                hidden_layer, output_layer, test, test_indices);

            const double average_epoch_loss =
                epoch_loss_sum / training_count;
            total_training_loss += epoch_loss_sum;

            std::cout << std::fixed << std::setprecision(6)
                      << epoch << ','
                      << average_epoch_loss << ','
                      << after.average_loss << ','
                      << after.accuracy * 100.0 << ','
                      << training_elapsed_seconds << '\n';

            if (time_to_target_seconds < 0.0 &&
                after.accuracy >= target_accuracy) {
                time_to_target_seconds = training_elapsed_seconds;
            }
        }

        std::cout << "Implementation: sequential DenseLayer model\n";
        std::cout << "Training examples per epoch: " << training_count << '\n';
        std::cout << "Epochs: " << epochs << '\n';
        std::cout << "Learning rate: " << learning_rate << '\n';
        std::cout << "Seed: " << seed << '\n';
        std::cout << "Test loss before training: "
                  << before.average_loss << '\n';
        std::cout << "Test accuracy before training: "
                  << before.accuracy * 100.0 << "%\n";
        std::cout << "Test loss after training: "
                  << after.average_loss << '\n';
        std::cout << "Test accuracy after training: "
                  << after.accuracy * 100.0 << "%\n";
        std::cout << "Total training time: "
                  << training_elapsed_seconds << " seconds\n";

        if (time_to_target_seconds >= 0.0) {
            std::cout << "Epoch-checkpoint time to 90% accuracy: "
                      << time_to_target_seconds << " seconds\n";
        } else {
            std::cout << "90% accuracy was not reached within "
                      << epochs << " epochs\n";
        }

        std::cout << "Total average training loss: "
                  << total_training_loss / (training_count * epochs) << '\n';
        std::cout << "Hidden-layer updates: "
                  << hidden_layer.version() << '\n';
        std::cout << "Output-layer updates: "
                  << output_layer.version() << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
#include "bounded_queue.h"
#include "dense_layer.h"
#include "mnist.h"
#include "messages.h"
#include "worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct WorkerLifetime {
    BoundedQueue<HiddenWorkMsg>& work;
    BoundedQueue<ActivationMsg>& activations;
    BoundedQueue<BatchResultMsg>& results;
    std::thread& hidden;
    std::thread& output;

    ~WorkerLifetime() {
        work.close();
        activations.close();
        results.close();
        if (hidden.joinable()) hidden.join();
        if (output.joinable()) output.join();
    }
};

struct Evaluation {
    double average_loss = 0.0;
    double accuracy = 0.0;
};

struct LossResult {
    float loss;
    std::size_t prediction;
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

LossResult calculate_loss_and_prediction(
    const std::vector<float>& logits,
    std::size_t correct_label) {
    if (logits.empty() || correct_label >= logits.size()) {
        throw std::runtime_error("Invalid logits or label");
    }

    float max_logit = logits[0];
    std::size_t prediction = 0;

    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            prediction = i;
        }
    }

    double sum_exp = 0.0;
    for (float logit : logits) {
        sum_exp += std::exp(static_cast<double>(logit - max_logit));
    }

    const float loss =
        max_logit + static_cast<float>(std::log(sum_exp))
        - logits[correct_label];

    return {loss, prediction};
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
        const LossResult prediction =
            calculate_loss_and_prediction(logits, data.labels[image_index]);

        result.average_loss += prediction.loss;
        if (prediction.prediction == data.labels[image_index]) {
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
        if (argc < 2 || argc > 4) {
            std::cerr
                << "Usage: ./mnist_reader.exe <tau_max> "
                << "[max_in_flight] [epochs]\n"
                << "Allowed tau_max values: 0, 1, 2, 4, 8, 16\n"
                << "Allowed max_in_flight values: 1, 2, 4, 8, 16\n"
                << "Epochs must be between 1 and 20\n";
            return 2;
        }

        const std::string tau_argument = argv[1];
        std::size_t tau_characters = 0;
        const std::uint64_t tau_max =
            std::stoull(tau_argument, &tau_characters);

        if (tau_characters != tau_argument.size()) {
            throw std::invalid_argument("tau_max must be an integer");
        }

        const std::vector<std::uint64_t> allowed_tau_values = {
            0, 1, 2, 4, 8, 16
        };

        if (std::find(allowed_tau_values.begin(),
                      allowed_tau_values.end(),
                      tau_max) == allowed_tau_values.end()) {
            throw std::invalid_argument(
                "tau_max must be one of: 0, 1, 2, 4, 8, 16");
        }

        std::size_t max_in_flight = 8;
        if (argc >= 3) {
            const std::string in_flight_argument = argv[2];
            std::size_t in_flight_characters = 0;
            max_in_flight = static_cast<std::size_t>(
                std::stoull(in_flight_argument, &in_flight_characters));

            if (in_flight_characters != in_flight_argument.size()) {
                throw std::invalid_argument(
                    "max_in_flight must be an integer");
            }
        }

        const std::vector<std::size_t> allowed_in_flight_values = {
            1, 2, 4, 8, 16
        };

        if (std::find(allowed_in_flight_values.begin(),
                      allowed_in_flight_values.end(),
                      max_in_flight) == allowed_in_flight_values.end()) {
            throw std::invalid_argument(
                "max_in_flight must be one of: 1, 2, 4, 8, 16");
        }

        std::size_t epochs = 1;
        if (argc == 4) {
            const std::string epochs_argument = argv[3];
            std::size_t epoch_characters = 0;
            epochs = static_cast<std::size_t>(
                std::stoull(epochs_argument, &epoch_characters));

            if (epoch_characters != epochs_argument.size()) {
                throw std::invalid_argument("epochs must be an integer");
            }
        }

        if (epochs < 1 || epochs > 20) {
            throw std::invalid_argument(
                "epochs must be between 1 and 20");
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

        if (train.labels.size() < training_count ||
            test.labels.size() < evaluation_count) {
            throw std::runtime_error(
                "Dataset is smaller than requested subset");
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

        BoundedQueue<HiddenWorkMsg> work_queue(max_in_flight);
        BoundedQueue<ActivationMsg> activation_queue(max_in_flight);
        BoundedQueue<BatchResultMsg> result_queue(max_in_flight);

        HiddenWorker hidden_worker(
            hidden_layer,
            work_queue,
            activation_queue,
            result_queue,
            learning_rate,
            tau_max);

        OutputWorker output_worker(
            output_layer,
            activation_queue,
            work_queue,
            learning_rate);

        std::thread output_thread;
        std::thread hidden_thread;
        WorkerLifetime lifetime{work_queue, activation_queue, result_queue,
                                hidden_thread, output_thread};
        output_thread = std::thread(&OutputWorker::run, &output_worker);
        hidden_thread = std::thread(&HiddenWorker::run, &hidden_worker);

        std::size_t total_completed = 0;
        std::size_t total_accepted = 0;
        std::size_t total_discarded = 0;
        std::uint64_t total_gradient_age = 0;
        std::uint64_t accepted_gradient_age_sum = 0;
        std::uint64_t maximum_gradient_age = 0;
        double total_training_loss = 0.0;
        double cumulative_training_seconds = 0.0;

        std::vector<Evaluation> epoch_evaluations;
        std::vector<double> epoch_average_losses;
        std::vector<double> epoch_elapsed_seconds;
        epoch_evaluations.reserve(epochs);
        epoch_average_losses.reserve(epochs);
        epoch_elapsed_seconds.reserve(epochs);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(train_indices.begin(), train_indices.end(), generator);

            std::vector<bool> completed(training_count, false);
            std::size_t submitted = 0;
            std::size_t completed_this_epoch = 0;
            double epoch_training_loss = 0.0;

            const std::uint64_t batch_id_offset =
                static_cast<std::uint64_t>(epoch) * training_count;

            const auto epoch_training_start =
                std::chrono::steady_clock::now();

            while (completed_this_epoch < training_count) {
                while (submitted < training_count &&
                       submitted - completed_this_epoch < max_in_flight) {
                    const std::size_t image_index = train_indices[submitted];

                    TrainingMsg training;
                    training.batch_id = batch_id_offset + submitted;
                    training.label = train.labels[image_index];
                    training.input = make_input(train, image_index);
                    training.enqueued_at =
                        std::chrono::steady_clock::now();

                    if (!work_queue.push(
                            HiddenWorkMsg{std::move(training)})) {
                        break;
                    }

                    ++submitted;
                }

                BatchResultMsg result;
                if (!result_queue.pop(result)) {
                    throw std::runtime_error(
                        "Result queue closed before the epoch completed");
                }

                if (result.batch_id < batch_id_offset ||
                    result.batch_id >= batch_id_offset + training_count) {
                    throw std::runtime_error(
                        "Coordinator received a result from the wrong epoch");
                }

                const std::size_t local_batch_id =
                    static_cast<std::size_t>(
                        result.batch_id - batch_id_offset);

                if (completed[local_batch_id]) {
                    throw std::runtime_error(
                        "Coordinator received a duplicate batch result");
                }

                completed[local_batch_id] = true;
                ++completed_this_epoch;
                ++total_completed;

                epoch_training_loss += result.loss;
                total_training_loss += result.loss;
                total_gradient_age += result.gradient_age;
                maximum_gradient_age =
                    std::max(maximum_gradient_age, result.gradient_age);

                if (result.gradient_accepted) {
                    ++total_accepted;
                    accepted_gradient_age_sum += result.gradient_age;
                } else {
                    ++total_discarded;
                }
            }

            const auto epoch_training_end =
                std::chrono::steady_clock::now();
            cumulative_training_seconds +=
                std::chrono::duration<double>(
                    epoch_training_end - epoch_training_start).count();

            epoch_average_losses.push_back(
                epoch_training_loss / training_count);
            epoch_elapsed_seconds.push_back(cumulative_training_seconds);

            epoch_evaluations.push_back(
                evaluate(hidden_layer, output_layer, test, test_indices));
        }

        work_queue.close();
        hidden_thread.join();
        output_thread.join();

        hidden_worker.rethrow_if_error();
        output_worker.rethrow_if_error();

        const std::size_t hidden_updates =
            static_cast<std::size_t>(hidden_layer.version());
        const std::size_t output_updates =
            static_cast<std::size_t>(output_layer.version());

        const WorkerMetrics& hidden_metrics = hidden_worker.metrics();
        const WorkerMetrics& output_metrics = output_worker.metrics();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "epoch,average_training_loss,test_loss,"
                     "test_accuracy_percent,elapsed_training_seconds\n";

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            const Evaluation& evaluation = epoch_evaluations[epoch];

            std::cout << (epoch + 1) << ','
                      << epoch_average_losses[epoch] << ','
                      << evaluation.average_loss << ','
                      << evaluation.accuracy * 100.0 << ','
                      << epoch_elapsed_seconds[epoch] << '\n';
        }

        std::cout << "Implementation: bounded delayed-gradient model\n";
        std::cout << "Workers: 2\n";
        std::cout << "Maximum batches in flight: " << max_in_flight << '\n';
        std::cout << "Maximum gradient age: " << tau_max << '\n';
        std::cout << "Training examples per epoch: "
                  << training_count << '\n';
        std::cout << "Completed examples: " << total_completed << '\n';
        std::cout << "Epochs: " << epochs << '\n';
        std::cout << "Learning rate: " << learning_rate << '\n';
        std::cout << "Seed: " << seed << '\n';
        std::cout << "Average training loss: "
                  << total_training_loss / total_completed << '\n';
        std::cout << "Worker 0 accepted updates: "
                  << hidden_updates << '\n';
        std::cout << "Worker 0 discarded gradients: "
                  << total_discarded << '\n';
        std::cout << "Worker 0 update acceptance: "
                  << 100.0 * hidden_updates / total_completed << "%\n";
        std::cout << "Worker 1 updates: " << output_updates << '\n';
        std::cout << "Average age of all gradients: "
                  << static_cast<double>(total_gradient_age) /
                         total_completed << '\n';
        std::cout << "Average age of accepted gradients: "
                  << (total_accepted == 0
                          ? 0.0
                          : static_cast<double>(
                                accepted_gradient_age_sum) /
                                total_accepted)
                  << '\n';
        std::cout << "Maximum observed gradient age: "
                  << maximum_gradient_age << '\n';
        std::cout << "Test subset size: " << evaluation_count << '\n';
        std::cout << "Test loss before training: "
                  << before.average_loss << '\n';
        std::cout << "Test accuracy before training: "
                  << before.accuracy * 100.0 << "%\n";
        std::cout << "Test loss after training: "
                  << epoch_evaluations.back().average_loss << '\n';
        std::cout << "Test accuracy after training: "
                  << epoch_evaluations.back().accuracy * 100.0 << "%\n";
        std::cout << "Total training time: "
                  << cumulative_training_seconds << " seconds\n";
        std::cout << "Completed-example rate: "
                  << total_completed / cumulative_training_seconds
                  << " examples/second\n";
        std::cout << "Worker 0 update rate: "
                  << hidden_updates / cumulative_training_seconds
                  << " updates/second\n";
        std::cout << "Worker 1 update rate: "
                  << output_updates / cumulative_training_seconds
                  << " updates/second\n";

        std::cout << "Worker 0 processed batches: "
                  << hidden_metrics.processed_batches << '\n';
        std::cout << "Worker 0 active-path time "
                     "(processing and queue pushes): "
                  << hidden_metrics.active_seconds << " seconds\n";
        std::cout << "Worker 0 input-queue wait time: "
                  << hidden_metrics.waiting_seconds << " seconds\n";
        std::cout << "Worker 1 processed batches: "
                  << output_metrics.processed_batches << '\n';
        std::cout << "Worker 1 active-path time "
                     "(processing and queue pushes): "
                  << output_metrics.active_seconds << " seconds\n";
        std::cout << "Worker 1 input-queue wait time: "
                  << output_metrics.waiting_seconds << " seconds\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
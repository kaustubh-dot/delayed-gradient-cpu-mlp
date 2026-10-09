#include "worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

struct LossAndGradient {
    float loss;
    std::vector<float> output_gradient;
};

LossAndGradient calculate_loss_and_gradient(
    const std::vector<float>& logits,
    std::size_t correct_label) {
    if (logits.empty() || correct_label >= logits.size()) {
        throw std::runtime_error("Invalid logits or label in output worker");
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

} // namespace

HiddenWorker::HiddenWorker(
    DenseLayer& layer,
    BoundedQueue<HiddenWorkMsg>& work_queue,
    BoundedQueue<ActivationMsg>& activation_queue,
    BoundedQueue<BatchResultMsg>& result_queue,
    float learning_rate,
    std::uint64_t tau_max)
    : layer_(layer),
      work_queue_(work_queue),
      activation_queue_(activation_queue),
      result_queue_(result_queue),
      learning_rate_(learning_rate),
      tau_max_(tau_max) {}

void HiddenWorker::run() {
    std::unordered_map<std::size_t, std::vector<float>> pending_inputs;

    try {
        HiddenWorkMsg work;

        while (true) {
            const auto wait_start = Clock::now();
            const bool received_work = work_queue_.pop(work);
            metrics_.waiting_seconds += seconds_between(wait_start, Clock::now());

            if (!received_work) {
                break;
            }

            if (TrainingMsg* training = std::get_if<TrainingMsg>(&work)) {
                const auto work_start = Clock::now();

                const auto inserted =
                    pending_inputs.emplace(training->batch_id, training->input);
                if (!inserted.second) {
                    throw std::runtime_error(
                        "Hidden worker received a duplicate batch ID");
                }

                ActivationMsg activation;
                activation.batch_id = training->batch_id;
                activation.label = training->label;
                activation.source_version = layer_.version();
                activation.values = layer_.forward(training->input);
                activation.enqueued_at = Clock::now();

                if (!activation_queue_.push(std::move(activation))) {
                    break;
                }

                metrics_.active_seconds +=
                    seconds_between(work_start, Clock::now());
                continue;
            }

            GradientMsg& gradient = std::get<GradientMsg>(work);
            const auto cache = pending_inputs.find(gradient.batch_id);

            if (cache == pending_inputs.end()) {
                throw std::runtime_error(
                    "Hidden worker has no cached input for a gradient");
            }
            if (gradient.origin_version > layer_.version()) {
                throw std::runtime_error(
                    "Gradient origin version is newer than the hidden layer");
            }

            const auto work_start = Clock::now();
            const std::uint64_t age =
                layer_.version() - gradient.origin_version;
            const bool accepted = age <= tau_max_;

            if (accepted) {
                const LayerGradients gradients =
                    layer_.backward(cache->second, gradient.values);
                layer_.apply_gradients(gradients, learning_rate_);
                ++metrics_.accepted_gradients;
            } else {
                ++metrics_.discarded_gradients;
            }

            metrics_.maximum_gradient_age =
                std::max(metrics_.maximum_gradient_age, age);
            metrics_.total_gradient_age += age;

            BatchResultMsg result;
            result.batch_id = gradient.batch_id;
            result.loss = gradient.loss;
            result.gradient_age = age;
            result.gradient_accepted = accepted;
            result.worker0_version = layer_.version();
            result.worker1_version = gradient.producer_version + 1;

            pending_inputs.erase(cache);

            if (!result_queue_.push(std::move(result))) {
                break;
            }

            ++metrics_.processed_batches;
            metrics_.active_seconds +=
                seconds_between(work_start, Clock::now());
        }

        activation_queue_.close();
        result_queue_.close();
    } catch (...) {
        error_ = std::current_exception();
        work_queue_.close();
        activation_queue_.close();
        result_queue_.close();
    }
}

const WorkerMetrics& HiddenWorker::metrics() const {
    return metrics_;
}

void HiddenWorker::rethrow_if_error() const {
    if (error_) {
        std::rethrow_exception(error_);
    }
}

OutputWorker::OutputWorker(
    DenseLayer& layer,
    BoundedQueue<ActivationMsg>& activation_queue,
    BoundedQueue<HiddenWorkMsg>& work_queue,
    float learning_rate)
    : layer_(layer),
      activation_queue_(activation_queue),
      work_queue_(work_queue),
      learning_rate_(learning_rate) {}

void OutputWorker::run() {
    try {
        ActivationMsg activation;

        while (true) {
            const auto wait_start = Clock::now();
            const bool received_activation =
                activation_queue_.pop(activation);
            metrics_.waiting_seconds += seconds_between(wait_start, Clock::now());

            if (!received_activation) {
                break;
            }

            const auto work_start = Clock::now();

            const std::vector<float> logits =
                layer_.forward(activation.values);
            const LossAndGradient loss =
                calculate_loss_and_gradient(logits, activation.label);
            const std::uint64_t version_used = layer_.version();

            const LayerGradients gradients =
                layer_.backward(activation.values, loss.output_gradient);

            GradientMsg gradient;
            gradient.batch_id = activation.batch_id;
            gradient.origin_version = activation.source_version;
            gradient.producer_version = version_used;
            gradient.loss = loss.loss;
            gradient.values = gradients.input_gradient;
            gradient.enqueued_at = Clock::now();

            // Calculate the hidden-layer gradient before changing output weights.
            layer_.apply_gradients(gradients, learning_rate_);

            if (!work_queue_.push(HiddenWorkMsg{std::move(gradient)})) {
                break;
            }

            ++metrics_.processed_batches;
            metrics_.active_seconds +=
                seconds_between(work_start, Clock::now());
        }

        // Do not close the shared work queue; the coordinator also produces to it.
    } catch (...) {
        error_ = std::current_exception();
        activation_queue_.close();
        work_queue_.close();
    }
}

const WorkerMetrics& OutputWorker::metrics() const {
    return metrics_;
}

void OutputWorker::rethrow_if_error() const {
    if (error_) {
        std::rethrow_exception(error_);
    }
}
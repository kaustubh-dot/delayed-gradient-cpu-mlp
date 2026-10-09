#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

struct TrainingMsg {
    std::size_t batch_id = 0;
    std::size_t label = 0;
    std::vector<float> input;
    std::chrono::steady_clock::time_point enqueued_at;
};

struct ActivationMsg {
    std::size_t batch_id = 0;
    std::size_t label = 0;

    // Worker 0 version used to produce this activation.
    std::uint64_t source_version = 0;

    std::vector<float> values;
    std::chrono::steady_clock::time_point enqueued_at;
};

struct GradientMsg {
    std::size_t batch_id = 0;

    // Worker 0 version used to produce the activation for this batch.
    std::uint64_t origin_version = 0;

    // Worker 1 version used to calculate this backward gradient.
    std::uint64_t producer_version = 0;

    float loss = 0.0f;

    // Gradient of loss with respect to Worker 0's hidden activations.
    std::vector<float> values;
    std::chrono::steady_clock::time_point enqueued_at;
};

// Both the coordinator and Worker 1 can send work to Worker 0.
// Worker 0 processes these messages serially, as the sole owner of its layer.
using HiddenWorkMsg = std::variant<TrainingMsg, GradientMsg>;

struct BatchResultMsg {
    std::size_t batch_id = 0;
    float loss = 0.0f;

    // Age of the gradient when Worker 0 received it.
    std::uint64_t gradient_age = 0;

    // Whether Worker 0 applied or discarded this gradient.
    bool gradient_accepted = false;

    std::uint64_t worker0_version = 0;
    std::uint64_t worker1_version = 0;
};
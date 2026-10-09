#pragma once

#include "bounded_queue.h"
#include "dense_layer.h"
#include "messages.h"

#include <cstddef>
#include <cstdint>
#include <exception>

struct WorkerMetrics {
    double active_seconds = 0.0;
    double waiting_seconds = 0.0;
    std::size_t processed_batches = 0;
    std::size_t accepted_gradients = 0;
    std::size_t discarded_gradients = 0;
    std::uint64_t maximum_gradient_age = 0;
    std::uint64_t total_gradient_age = 0;
};

class HiddenWorker {
public:
    HiddenWorker(
        DenseLayer& layer,
        BoundedQueue<HiddenWorkMsg>& work_queue,
        BoundedQueue<ActivationMsg>& activation_queue,
        BoundedQueue<BatchResultMsg>& result_queue,
        float learning_rate,
        std::uint64_t tau_max);

    void run();
    const WorkerMetrics& metrics() const;
    void rethrow_if_error() const;

private:
    DenseLayer& layer_;
    BoundedQueue<HiddenWorkMsg>& work_queue_;
    BoundedQueue<ActivationMsg>& activation_queue_;
    BoundedQueue<BatchResultMsg>& result_queue_;
    float learning_rate_;
    std::uint64_t tau_max_;
    WorkerMetrics metrics_;
    std::exception_ptr error_;
};

class OutputWorker {
public:
    OutputWorker(
        DenseLayer& layer,
        BoundedQueue<ActivationMsg>& activation_queue,
        BoundedQueue<HiddenWorkMsg>& work_queue,
        float learning_rate);

    void run();
    const WorkerMetrics& metrics() const;
    void rethrow_if_error() const;

private:
    DenseLayer& layer_;
    BoundedQueue<ActivationMsg>& activation_queue_;
    BoundedQueue<HiddenWorkMsg>& work_queue_;
    float learning_rate_;
    WorkerMetrics metrics_;
    std::exception_ptr error_;
};
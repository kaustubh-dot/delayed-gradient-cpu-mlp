#include "bounded_queue.h"
#include "dense_layer.h"
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check_derivative(Activation activation) {
    DenseLayer layer(5, 4, activation, 17);
    std::vector<float> x{0.3f, -0.8f, 0.5f, 0.2f, -0.1f};
    std::vector<float> upstream{0.7f, -0.2f, 0.4f, -0.5f};
    const auto gradients = layer.backward(x, upstream);
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto plus = x, minus = x;
        const float epsilon = 0.001f;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        auto a = layer.forward(plus), b = layer.forward(minus);
        double numerical = 0;
        for (std::size_t j = 0; j < a.size(); ++j)
            numerical += upstream[j] * (a[j] - b[j]) / (2 * epsilon);
        require(std::abs(numerical - gradients.input_gradient[i]) < 0.0002,
                "Input derivative differs from finite differences");
    }
    const auto before = layer.forward(x);
    layer.apply_gradients(gradients, 0.01f);
    const auto after = layer.forward(x);
    double change = 0;
    for (std::size_t j = 0; j < before.size(); ++j)
        change += upstream[j] * (after[j] - before[j]);
    require(change < 0, "SGD failed to reduce the tested linear objective");
    require(layer.version() == 1, "Accepted update must increment version");
    bool rejected = false;
    try { layer.forward({1}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Wrong input dimensions were accepted");
}

void check_queue() {
    using namespace std::chrono_literals;
    BoundedQueue<int> queue(1);
    require(queue.push(7), "Initial push failed");
    auto producer = std::async(std::launch::async, [&] { return queue.push(8); });
    require(producer.wait_for(50ms) == std::future_status::timeout,
            "A full queue allowed a second item");
    queue.close();
    require(producer.wait_for(2s) == std::future_status::ready && !producer.get(),
            "Closure did not release a producer");
    int value = 0;
    require(queue.pop(value) && value == 7, "Closure must preserve queued data");
    require(!queue.pop(value), "Closed empty queue must report end of stream");
    BoundedQueue<int> empty(1);
    auto consumer = std::async(std::launch::async, [&] { int item; return empty.pop(item); });
    require(consumer.wait_for(50ms) == std::future_status::timeout,
            "Empty queue returned an item");
    empty.close();
    require(consumer.wait_for(2s) == std::future_status::ready && !consumer.get(),
            "Closure did not release a consumer");
}

int main() {
    try {
        check_derivative(Activation::None);
        check_derivative(Activation::Relu);
        check_queue();
        std::cout << "Dense derivatives, SGD versions and queue closure passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

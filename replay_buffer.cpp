#pragma once
#include <torch/torch.h>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>

struct Transition {
    torch::Tensor state;
    int action;
    float reward;
    torch::Tensor next_state;
    bool terminal;
};

class ReplayBuffer {
private:
    std::vector<Transition> buffer;
    size_t capacity;
    size_t position = 0;
    std::mt19937 rng;

public:
    ReplayBuffer(size_t capacity = 300000, uint32_t seed = std::random_device{}())
        : capacity(capacity), rng(seed) {
        buffer.reserve(capacity);
    }

    void push(torch::Tensor state, int action, float reward, torch::Tensor next_state, bool done) {
        if (buffer.size() < capacity) {
            buffer.push_back({state.clone(), action, reward, next_state.clone(), done});
        } else {
            buffer[position] = {state.clone(), action, reward, next_state.clone(), done};
            position = (position + 1) % capacity;
        }
    }

    std::vector<Transition> sample(size_t batch_size) {
        std::vector<Transition> batch;
        if (batch_size > buffer.size()) {
            throw std::invalid_argument("batch_size cannot exceed replay buffer size");
        }
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[dist(rng)]);
            
        }
        return batch;
    }

    size_t size() const { return buffer.size(); }
};

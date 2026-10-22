#pragma once
#include <torch/torch.h>
#include "replay_buffer.cpp"

struct QNetworkImpl : torch::nn::Module {
torch::nn::Linear fc1{nullptr} , fc2{nullptr} , out{nullptr};

    QNetworkImpl(int input_dim = 6 , int out_dim = 3) {
        fc1 = register_module("fc1" , torch::nn::Linear(input_dim , 128));
        fc2 = register_module("fc2" , torch::nn::Linear(128 , 128));
        out = register_module("out" , torch::nn::Linear(128 , out_dim));
        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::xavier_uniform_(fc2->weight);
        torch::nn::init::xavier_uniform_(out->weight);

        torch::nn::init::zeros_(fc1->bias);
        torch::nn::init::zeros_(fc2->bias);
        torch::nn::init::zeros_(out->bias);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return out->forward(x);
    }
};
TORCH_MODULE(QNetwork);


inline float backward(QNetwork& q_net , QNetwork& target_net , torch::optim::Adam& optimizer ,const std::vector<Transition>& batch , float gamma = 0.99f) {
    if (batch.empty()) return 0.0f;
    const auto feature_count = batch.front().state.numel();
    std::vector<float> states_data, next_states_data;
    std::vector<int64_t> actions_list;
    std::vector<float> rewards_list, dones_list;
    states_data.reserve(batch.size() * feature_count);
    next_states_data.reserve(batch.size() * feature_count);
    actions_list.reserve(batch.size());
    rewards_list.reserve(batch.size());
    dones_list.reserve(batch.size());

    for (const auto& t : batch) {
        auto state = t.state.contiguous().view({-1});
        auto next_state = t.next_state.contiguous().view({-1});
        states_data.insert(states_data.end(), state.data_ptr<float>(), state.data_ptr<float>() + feature_count);
        next_states_data.insert(next_states_data.end(), next_state.data_ptr<float>(), next_state.data_ptr<float>() + feature_count);
        actions_list.push_back(t.action);
        rewards_list.push_back(t.reward);
        dones_list.push_back(t.terminal ? 1.0f : 0.0f);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor states = torch::from_blob(states_data.data(), {static_cast<int64_t>(batch.size()), feature_count}, options).clone();
    torch::Tensor next_states = torch::from_blob(next_states_data.data(), {static_cast<int64_t>(batch.size()), feature_count}, options).clone();
    torch::Tensor actions = torch::tensor(actions_list, torch::kInt64);
    torch::Tensor rewards = torch::tensor(rewards_list, torch::kFloat32);
    torch::Tensor dones = torch::tensor(dones_list, torch::kFloat32);

    torch::Tensor qvalues = q_net->forward(states);
    torch::Tensor current_q = qvalues.gather(1 , actions.unsqueeze(1)).squeeze(1);

    torch::Tensor target_q;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor next_q_all = target_net(next_states);                 
        torch::Tensor max_next_q = std::get<0>(next_q_all.max(1));    
        target_q = rewards + (gamma * max_next_q * (1.0f - dones)); 
    }

    auto loss = torch::nn::functional::mse_loss(current_q , target_q);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    return loss.item<float>();

}

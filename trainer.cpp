#include <utility>
#include <iostream>
#include <filesystem>
#include "env.cpp"
#include "NN.cpp"
#include "replay_buffer.cpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <cstdlib>

struct TrainerOptions {
    int threads = 1;
    int matches = 600;
    int batch_size = 64;
    uint32_t seed = std::random_device{}();
    bool fresh = false;
    std::string checkpoint = "models/model.pt";
    std::string output = "models/model_600.pt";
};

static TrainerOptions parse_options(int argc, char** argv) {
    TrainerOptions options;
    options.threads = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--threads") options.threads = std::stoi(value("--threads"));
        else if (arg == "--matches") options.matches = std::stoi(value("--matches"));
        else if (arg == "--batch-size") options.batch_size = std::stoi(value("--batch-size"));
        else if (arg == "--seed") options.seed = static_cast<uint32_t>(std::stoul(value("--seed")));
        else if (arg == "--fresh") options.fresh = true;
        else if (arg == "--checkpoint") options.checkpoint = value("--checkpoint");
        else if (arg == "--output") options.output = value("--output");
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: trainer [--threads N] [--matches N] [--batch-size N] [--seed N] [--fresh] [--checkpoint PATH] [--output PATH]\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown option: " + arg);
    }
    if (options.threads < 1 || options.matches < 1 || options.batch_size < 1)
        throw std::invalid_argument("threads, matches, and batch-size must be positive");
    return options;
}

std::pair<int , torch::Tensor>  B_E(torch::Tensor q_values , double tua) {
    q_values = q_values.to(torch::kFloat32);
    torch::Tensor scaled_qvalues = q_values/tua;
    torch::Tensor qmax = torch::max(scaled_qvalues);
    torch::Tensor stable_qvalues = scaled_qvalues-qmax;
    torch::Tensor prob = torch::softmax(stable_qvalues , 0);
    torch::Tensor action = torch::multinomial(prob , 1);
    return {action.item<int>() , prob};
}

int main(int argc, char** argv) {
    TrainerOptions run;
    try { run = parse_options(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }

    
    torch::set_num_threads(run.threads);
    torch::set_num_interop_threads(1);
    torch::manual_seed(run.seed);
    std::srand(run.seed);
    std::cout << "CPU threads: " << run.threads << " | seed: " << run.seed << '\n';
    PPEnv env;
    QNetwork q_net(6 , 3);
    QNetwork target_net(6, 3);

    if (!run.fresh && std::filesystem::exists(run.checkpoint)) {
        try {
            torch::load(q_net, run.checkpoint);
            std::cout << "Loaded checkpoint from " << run.checkpoint << '\n';
        } catch (const std::exception& e) {
            std::cout << "Failed to load model checkpoint. Starting fresh.\n";
        }
    } else {
        std::cout << "Starting with Random weights\n";
    }   
    {
    torch::NoGradGuard no_grad;
    auto target_params = target_net->named_parameters();
    for (auto& param : q_net->named_parameters()) {
        target_params[param.key()].copy_(param.value());
    }
    }
    int capacity = 100000;
    int batch_size = run.batch_size;
    int min_buffer_size = 2000;
    int Num_matches = run.matches;
    constexpr int match_endpoint = 11;
    constexpr float match_win_reward = 5.0f;
    int total_steps = 0;
    int total_steps_BE = 0;
    int target_update_freq = 3000;
    // float epsilon_start = 1.0f;
    // float epsilon_min = 0.05f;
    // float decay_steps = 1100;
    // float progress = 0.0f;
    double tua_max = 2.0;
    double tua_min = 0.1;
    double  tua = tua_max;
    int decay_steps = 9000; 
    /* double tua_decay = std::pow(tua_min/tua_max , total_steps_BE/decay_length); */
    double progress;

    ReplayBuffer buffer(capacity, run.seed);
    double learning_rate = 1e-4;
    auto options = torch::optim::AdamOptions(learning_rate).weight_decay(1e-5);
    torch::optim::Adam optimizer(q_net->parameters() , options);
    torch::nn::MSELoss criterion;

    torch::Tensor state_l = env.get_side_observations(1);
    torch::Tensor state_r = env.get_side_observations(0);


    for(int i = 0 ; i < Num_matches ; i++) {
        env.left_score = 0;
        env.right_score = 0;
        while(env.left_score < 11 && env.right_score < 11){
        int action_r;
        int action_l;

        total_steps++;
        total_steps_BE++;
        
            //epsilon greedy policy balances exploration and exploitation
        // if(dist(rng) < epsilon) action_r = action_dist(rng);
        // else 
            // progress = std::min(1.0f , static_cast<float>(total_steps_E/decay_steps));
            // epsilon = epsilon_start+progress*(epsilon_min-epsilon_start);
        std::pair<int , torch::Tensor> a;
        {
        torch::NoGradGuard no_grad;
        torch::Tensor q_vals = q_net(state_r.unsqueeze(0));
        a = B_E(q_vals.squeeze(0) , tua);
        }
        action_r = a.first;
        // Both players use the shared policy from their own side-relative state.
        // Training from both perspectives makes this actual self-play.
        {
            torch::NoGradGuard no_grad;
            torch::Tensor q_vals = q_net(state_l.unsqueeze(0));
            action_l = B_E(q_vals.squeeze(0), tua).first;
        }

        const torch::Tensor previous_state_l = state_l;
    

        progress = std::min(static_cast<double>(1) , static_cast<double>(total_steps_BE)/decay_steps);
        tua = tua_max + progress*(tua_min-tua_max); 
        
        // float target_y = env.ball.y+opp_err(rng);
        // if(env.pleft.y+(env.pleft.height/2) < target_y) action_l = (1);
        // else if(env.pleft.y+(env.pleft.height/2) > target_y) action_l = (2);
        // else action_l = 0;
        StepResult result_r , result_l;
        env.action_update(action_r , action_l , result_r , result_l); 

        const bool match_ended = result_r.done &&
            (env.right_score >= match_endpoint || env.left_score >= match_endpoint);
        if (match_ended) {
            result_r.reward += (env.right_score >= match_endpoint)
                ? match_win_reward : -match_win_reward;
            result_l.reward -= (env.right_score >= match_endpoint)
                ? match_win_reward : -match_win_reward;
        }

        if (result_r.done && !match_ended) {
            env.reset_ball();
            state_l = env.get_side_observations(1);
            result_r.state = env.get_side_observations(0);
        } else {
            state_l = result_l.state;
        }
        if (result_r.done && !match_ended) {
            result_l.state = state_l;
        }
        buffer.push(previous_state_l, action_l,
                    result_l.reward, result_l.state, match_ended);
        buffer.push(state_r, action_r, result_r.reward, result_r.state, match_ended);
        state_r = result_r.state;
        state_l = result_l.state;

            if(buffer.size() >= min_buffer_size) {
                std::vector<Transition> batch = buffer.sample(batch_size);
                float loss = backward(q_net ,target_net ,  optimizer , batch , 0.99f);
                //std::cout<<"Backward Pass Loss : "<<loss<<"\n";
            }
        
        if (total_steps % target_update_freq == 0) {
                torch::NoGradGuard no_grad_sync;
                auto t_params = target_net->named_parameters();
                for (auto& p : q_net->named_parameters()) {
                    t_params[p.key()].copy_(p.value());
                }
                // Keep total_steps monotonic; the modulo schedules future updates.
            }
        //env.renderer();
           // std::cout<<a.second<<"\n"; probabilities rendering
        }
        std::cout<<"MATCH "<<i+1<<"/"<<Num_matches<<'\n';
        std::cout<<"AGENT SCORE : "<<env.right_score<<"\n";
        std::cout<<"OPPONENT SCORE : "<<env.left_score<<"\n";
        env.Reset_game();
        state_l = env.get_side_observations(1);
        state_r = env.get_side_observations(0); 
    }

    const std::filesystem::path output_path(run.output);
    if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
    torch::save(q_net, run.output);
    std::cout << "Successfully saved model weights to " << run.output << '\n';
    return 0;

}

#include <iostream>
#include "env.cpp"
#include "NN.cpp"
#include <filesystem>
int main() {
    InitWindow(Screen_width , Screen_height , "PING PONG");
    SetTargetFPS(60);


    PPEnv env;
    QNetwork q_net(6 , 3);
    if (std::filesystem::exists("models/model.pt")) {
        try {
            torch::load(q_net, "models/model.pt");
            std::cout << "Loaded checkpoint from model.pt\n";
        } catch (const std::exception& e) {
            std::cout << "Failed to load model checkpoint. Starting fresh.\n";
        }
    } else {
        std::cout << "Starting with Random weights\n";
    }   
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<float> opp_err(-100.0f , 100.0f);
    torch::Tensor state = env.get_side_observations(0);
    while(!WindowShouldClose() && env.right_score < 11 && env.left_score < 11){
        int action;
        int action_l;
        {
        torch::NoGradGuard no_grad;
        torch::Tensor q_vals = q_net(state.unsqueeze(0));
        action = q_vals.argmax(1).item<int>();
        }
        float target_y = env.ball.y+opp_err(rng);
        if(env.pleft.y+(env.pleft.height/2) < target_y) action_l = (1);
        else if(env.pleft.y+(env.pleft.height/2) > target_y) action_l = (2);
        else action_l = 0; 
        StepResult result;
        env.action_update(action , action_l , result , result);
        state = result.state;

        if(result.done) {
            env.reset_ball();
            state = env.get_side_observations(0);
        }

        env.renderer();
        
    }
    std::cout<<"MATCH INFO\n";
    std::cout<<"AGENT SCORE : "<<env.right_score<<"\n";
    std::cout<<"YOUR SCORE : "<<env.left_score<<"\n";

    CloseWindow();
    return 0;
}

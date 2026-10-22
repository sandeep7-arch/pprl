#pragma once
#include <raylib.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>

const int Screen_width = 1280;
const int Screen_height = 800;

class Ball {
public:
    float x,y;
    float speed_x , speed_y;
    int radius;
    void draw() {
        DrawCircle(x , y , radius , WHITE);
    }
    void update() {
        x += speed_x;
        y += speed_y;

        if(y+radius >= Screen_height) {
            y = Screen_height-radius;
            speed_y = -std::abs(speed_y);
        }
        else if(y-radius <= 0) {
            y = 0+radius;
            speed_y = std::abs(speed_y);
        }
    }
};

class paddle {
public:
    float x,y;
    float speed_y;
    int width,height;
    void draw(){
        DrawRectangle(x , y , width , height , WHITE);
    }
    void update(int d){
        if(d==1) {
            y += speed_y;
        }
        else if(d==2) y -= speed_y;

        if(y<=0) y=0;
        else if(y+height >= Screen_height) y = Screen_height - height;
    }
    
};

struct StepResult {
    torch::Tensor state;
    float reward;
    bool done;
};

class PPEnv{
public:
    Ball ball;
    int right_score;
    int left_score;
    paddle pleft;
    paddle pright;


    PPEnv() { 
        ball.radius = 10;
        pleft.x = 10;
        pleft.y = Screen_height/2.0f-60;
        pleft.width = 25;
        pleft.height = 120;
        pleft.speed_y = 9.0f;

        pright.x = Screen_width-35;
        pright.y = Screen_height/2.0f-60;
        pright.width = 25;
        pright.height = 120;
        pright.speed_y = 9.0f;
        right_score = 0 ;
        left_score = 0;;
        Reset_game();
    } //constructor for the class
    torch::Tensor get_side_observations(int side) {
        //paddle relative features  , confusing as fk bro
        float self_y = (side == 1) ? pleft.y+(pleft.height)/2 : pright.y+(pright.height)/2;
        float self_x = (side == 1) ? pleft.x : pright.x;
        float opp_y  = (side == 1) ? pright.y+(pright.height)/2 : pleft.y+(pleft.height)/2;
        float balldX = (side==0) ? (self_x-ball.x) : (ball.x-self_x);
        float balldY = (self_y-ball.y);
        float speed_dX = (side==0) ? (-ball.speed_x) : (ball.speed_x);
            return torch::tensor({
                balldX/(float)Screen_width,
                balldY/(float)Screen_height,
                speed_dX/15.0f,
                ball.speed_y/15.0f,
                self_y/(float)Screen_height,
                opp_y/(float)Screen_height
            } , torch::kFloat32);
        }
    void  reset_ball() {
        
            ball.x = Screen_width/2.0f;
            ball.y = Screen_height/2.0f;
            ball.speed_x = (rand()%2) ? 7.0f : -7.0f;
            ball.speed_y = (rand()%2)?7.0f:-7.0f;
            
        }

    void Reset_game(){
        ball.radius = 10;
        ball.x = Screen_width/2.0f;
        ball.y = Screen_height/2.0f;
        ball.speed_x = (rand()%2) ? 7.0f : -7.0f;
        ball.speed_y = (rand()%2) ? 7.0f : -7.0f;

         pleft.x = 10;
         pleft.y = Screen_height/2.0f-60;
         pleft.width = 25;
         pleft.height = 120;
         pleft.speed_y = 9.0f;

         pright.x = Screen_width-35;
         pright.y = Screen_height/2.0f-60;
         pright.width = 25;
         pright.height = 120;
         pright.speed_y = 9.0f;
         right_score = 0 , left_score = 0;

    }

     void action_update (int action_right , int action_left , StepResult& res_r ,  StepResult& res_l) {
            //this is our model controlled by action and policy
        
            if(action_right == 1) pright.update(1);
            else if(action_right == 2) pright.update(2);
        
            
            // this is the left paddle which is controlled by us or some configurable AI
           // current one is a ball following paddle this can be changed by us later

        if(action_left == 1) pleft.update(1);
        else if(action_left == 2) pleft.update(2);

        // if(IsKeyDown(KEY_Q)) pleft.update(2); //this is manual control of left paddle
        // else if(IsKeyDown(KEY_A)) pleft.update(1);
        
        
        
            ball.update();

            float reward_r = 0.0f;
            float reward_l = 0.0f;
            bool done = false;
            const float max_speed = 10.0f;
            if(ball.speed_x > 0 && CheckCollisionCircleRec(Vector2{ball.x , ball.y},(float)ball.radius , Rectangle{pright.x , pright.y , (float)pright.width , (float)pright.height})){
            float paddle_center_R = pright.y+(pright.height/2);
               float hit_offset = (ball.y-paddle_center_R)/pright.height;
               hit_offset = std::clamp(hit_offset , -1.0f , 1.0f);
               ball.speed_y = hit_offset*max_speed;
               ball.x = pright.x-ball.radius;
               ball.speed_x = -std::abs(ball.speed_x)*1.05;
               ball.speed_x = std::clamp(ball.speed_x , -15.0f , 15.0f);
               reward_r += 0.1f;
            }
            if(ball.speed_x < 0 && CheckCollisionCircleRec(Vector2{ball.x , ball.y}, (float)ball.radius , Rectangle{pleft.x , pleft.y , (float)pleft.width , (float)pleft.height})){
            float paddle_center_L = pleft.y+(pleft.height/2);
                float hit_offset = (ball.y-paddle_center_L)/pleft.height;
                hit_offset = std::clamp(hit_offset , -1.0f , 1.0f);
                ball.speed_y = hit_offset*max_speed;

                ball.x = pleft.x + pleft.width + ball.radius;
                ball.speed_x = std::abs(ball.speed_x) * 1.05f;
                ball.speed_x = std::clamp(ball.speed_x , -15.0f , 15.0f);
                reward_l += 0.1f;         
        }

            if(ball.x > Screen_width) {
                reward_l += 1.0f;
                reward_r -= 1.0f;
                done = true;
                left_score++;
                
            }
            else if(ball.x < 0) {
                reward_r += 1.0f;
                reward_l -= 1.0f;
                done = true;
                right_score++;
            }

            
            res_l = {get_side_observations(1) , reward_l , done};
            res_r = {get_side_observations(0) , reward_r , done};

        }
        void renderer() {
            BeginDrawing();
            ClearBackground(BLACK);
            DrawLine(Screen_width/2 , 0 , Screen_width/2 , Screen_height , WHITE);
            ball.draw();
            pleft.draw();
            pright.draw();
            EndDrawing();

        }
};



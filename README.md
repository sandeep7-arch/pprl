



This is mini - project to learn fundamentals of RL which is a fundamental
concept for my main project 

NEURAL LAYERS USE 6 features , with two hidden layers(6x128x128x3)

This is a Deep Q-Network

And for training , initially it is trained by standard ball following
AI(ofcourse with some randomness to make that model make msitakes) ,  then
it will be trained by selfplay against itself(hope I will)

Initially , the balance between exploration of actions and exploitation is done
by epsilon greedy algo later changed to Boltzmann(softmax) algo

The backward pass has two different neural networks one which is used to
evaluate the Q-values , the target network , this network is not dynamic as the
main network , every few steps , the target network updates itself from the
original network

USED LIBRARIES:
libtorch , 
raylib(for graphics)

## CPU training

Build the trainer in Release mode for optimized CPU execution:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target trainer -j
```

Run it with configurable LibTorch CPU threads and run length:

```sh
./build/trainer --threads 1 --matches 300 --batch-size 64 --seed 1234 \
  --fresh --output models/selfplay_300.pt
```

`--threads` controls LibTorch's intra-op CPU pool. This network is small, so
more threads are not always faster; USE `--threads 1`, `2,` The default uses the host's reported hardware
concurrency. `--matches` and `--batch-size` can be adjusted for a shorter run
or a larger training batch.

Both paddles use the same current Q-network from their own side-relative
observations, and replay receives transitions from both sides. Each side gets
a `+5` reward for winning and `-5` for losing. `--fresh` ignores the default
checkpoint and starts with random weights. A point ending the rally is not
marked terminal in replay: the trainer resets the ball and stores that
next-rally state, so the DQN can bootstrap across rallies. Replay's `terminal`
flag is true only after the match-ending point, which prevents bootstrapping
into the next match.

https://github.com/user-attachments/assets/b7765057-52f2-4daf-9b12-49511edfb370 

THIS IS JUST 900 games of self play from random weights , sure it is learning



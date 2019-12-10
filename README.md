## IE 534 Assignment: Reinforcement Learning
## Final Project

#### Getting Started
You can either:

* This is an Pytorch implementation on Distributional Q-Learning
  The link to the paper is: https://arxiv.org/pdf/1707.06887.pdf


* The code is modified from a HW assignment from UIUC IE 534:
```bash
git clone https://github.com/mikuhatsune/ie534_rl_hw.git
```

Example training logs [example_solution/log_breakout_dqn.txt](example_solution/log_breakout_dqn.txt), and [example_solution/log_breakout_a2c.txt](example_solution/log_breakout_a2c.txt).
Format:
```
iter: iteration
n_ep: number of episodes (games played)
ep_len: running averaged episode length
ep_rew: running averaged episode clipped reward
raw_ep_rew: running averaged raw episode reward (actual raw game score)
env_step: number of environment simulation steps
time, rem: time passed, estimated time remain

iter    500 |loss   0.00 |n_ep    28 |ep_len   31.3 |ep_rew  -0.22 |raw_ep_rew   1.76 |env_step   1000 |time 00:04 rem 281:49
```

#### Important
Run these commands once to make BlueWaters happy (install a newer version of gym):
```bash
module load python/2.0.0
pip install gym[atari]==0.14 --user
```

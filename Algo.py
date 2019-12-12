# RL algorithms: DQN and simplified Actor-Critic

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import model, Replay
import random
from collections import namedtuple
import numpy as np
import torch.autograd as autograd


class C51:
    def __init__(self, obs_space, act_space, lr=1e-4, replay_size=500000, batch_size=32,
                 discount=0.99, target_update=2500, eps_decay=500000, device=None):
        self.obs_space, self.act_space = obs_space, act_space
        self.batch_size, self.discount, self.target_update = batch_size, discount, target_update
        self.device = device

        #Parameters from paper
        self.batch_size = batch_size
        self.n_atoms = 51
        self.Vmin = -10.0
        self.Vmax = 10.0
        self.dz = (self.Vmax - self.Vmin) / float(self.n_atoms - 1)
        self.z = torch.arange(self.Vmin, self.Vmax + self.dz, self.dz, device = device)
        if len(obs_space.shape) == 3:
            self.q_func = model.Model(n_in=obs_space.shape, n_out=act_space.n).to(device)
            self.target_q_func = model.Model(n_in=obs_space.shape, n_out=act_space.n).to(device)
            self.state_dtype = torch.uint8
        else:
            print ("observation shape not supported:", obs_space.shape)
            raise
        self.q_func.train()
        self.target_q_func.train()

        print ('parameters to optimize:',
            [(name, p.shape, p.requires_grad) for name,p in self.q_func.named_parameters()],
            '\n')
        self.optimizer = torch.optim.RMSprop(self.q_func.parameters(), lr=lr)
        #self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=lr, betas=(0.9,0.99), eps=1e-8)

        # number of action steps done
        self.num_act_steps = 0
        #self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.01, eps_decay
        self.eps_decay = eps_decay
        self.num_train_steps = 0
        self.double_q = True

        self.replay = Replay.NaiveReplay(replay_size)

    def compute_epsilon(self):
        # linearly anneal from 1 to 0.01 in eps_decay steps
        eps = 1.0 + (0.01 - 1) * self.num_act_steps / self.eps_decay
        eps = max(0.01, eps)

        return eps

    def act(self, obses):
        obses = torch.as_tensor(obses, device=self.device, dtype=self.state_dtype)
        eps = self.compute_epsilon()
        self.num_act_steps += 1

        with torch.no_grad():
            # print("#######")
            # print(self.q_func(obses))
            # print(self.q_func(obses).max(1))
            # print(self.q_func(obses).max(1)[1])
            greedy_actions = self.q_func(obses)[1].max(1)[1].tolist()


        actions = []
        for i in range(len(obses)):
            if random.random() < eps:
                a = random.randrange(self.act_space.n)
            else:
                a = greedy_actions[i]
            actions.append(a)
        return actions


    def observe(self, obses, actions, transitions):
        for s,a,(sn,r,t,_) in zip(obses, actions, transitions):
            if t: # (s,a) leads to a terminal state
                sn = None
            self.replay.add((s,a,r,sn))
        if len(self.replay) > self.batch_size and self.replay.cur_batch is None:
            self.replay.sample_torch(self.batch_size, self.device)

    def train(self):
        #[state_batch, action_batch, reward_batch, next_states, done] = self.replay.cur_batch
        state_batch, action_batch, reward_batch, done, next_states = self.replay.cur_batch
        self.replay.cur_batch = None

        # dist_list, prob = self.q_func(state_batch)       
        # q_values = np.sum(np.multiply(prob, self.z), axis=2) 
        
        # next_dist_list, next_prob = self.target_q_func(next_states).detach()
        # next_q_values = np.sum(np.multiply(prob_, self.z), axis=2) 



        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        # if self.double_q:
        #     next_state_values[~done] = next_q_values[
        #         torch.arange(len(next_q_values)), self.q_func(next_states).argmax(1)]                
        # else:
        #     next_state_values[~done] = next_q_values.max(1)[0]



        curr_dist, qvals  = self.q_func.forward(state_batch)
        # curr_action_dist = curr_dist[range(batch_size), actions]

        next_dist, next_qval = self.target_q_func.forward(next_states)
        next_qvals = next_qval.max(1)[1].tolist()

        #next_dist_ = next_dist[torch.arange(self.batch_size), action_batch]
        # next_actions = torch.max(next_qvals, 1)[1]
        # next_dist = self.model.softmax(next_dist)
        optimal_dist = next_dist.permute(1,0,2).cuda()

        # Get Optimal Actions for the next states (from distribution z)


        #m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        m_prob = torch.zeros((self.act_space.n, self.batch_size ,self.n_atoms), device=self.device)
        #m_prob = torch.zeros(self.batch_size, self.n_atoms)
        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(self.batch_size):
            if done[i]:
                Tz = reward_batch[i]
                Tz = torch.clamp(Tz, self.Vmin, self.Vmax)
                bj = (Tz - self.Vmin) / self.dz 
                m_l, m_u = torch.floor(bj).long(), torch.ceil(bj).long()
                m_prob[action_batch[i]][i][m_l] += (m_u - bj)
                m_prob[action_batch[i]][i][m_u] += (bj - m_l)
            else:
                for j in range(self.n_atoms):                   
                    Tz = reward_batch[i] + self.discount * self.z[j]
                    Tz = torch.clamp(Tz, self.Vmin, self.Vmax)
                    bj = (Tz - self.Vmin) / self.dz 
                    m_l, m_u = torch.floor(bj).long(), torch.ceil(bj).long()
                    m_prob[action_batch[i]][i][m_l] += optimal_dist[next_qvals[i]][i][j] * (m_u - bj)
                    m_prob[action_batch[i]][i][m_u] += optimal_dist[next_qvals[i]][i][j] * (bj - m_l)
            
        #loss = - torch.sum(optimal_dist * (torch.log(optimal_dist) - torch.log(m_prob)))

        loss = - F.kl_div(optimal_dist ,m_prob)/self.batch_size
        loss = loss.cuda()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), 10)
        self.optimizer.step()
        # The ugly patch for using Adam on BlueWaters...
        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         state = self.optimizer.state[p]
        #         if state['step'] >= 1022:
        #             state['step'] = 1022

        self.num_train_steps += 1
        if self.num_train_steps % self.target_update == 0:
            self.target_q_func.load_state_dict(self.q_func.state_dict())
        return loss.item()

    def save(self, path):
        torch.save([self.q_func.state_dict(), self.target_q_func.state_dict(), self.optimizer.state_dict()], path)

    def load(self, path):
        s1, s2, s3 = torch.load(path, map_location=self.device)
        self.q_func.load_state_dict(s1)
        self.target_q_func.load_state_dict(s2)
        self.optimizer.load_state_dict(s3)




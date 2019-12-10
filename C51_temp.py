import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import time
import numpy as np
from model import Model
import random
import Replay


class C51(object):
	def __init__(self, state_space, act_space, lr=1e-4, replay_size=50000, batch_size=32,
                 discount=0.99, target_update=2500, eps_decay=500000, device=None):		
		self.state_space = state_space
		self.discount = discount
		self.lr = lr
		self.batch_size = batch_size 
		self.target_update = target_update

		#Parameters from paper
		self.n_atoms = 51
        self.Vmin = -10.0
        self.Vmax = 10.0
        self.dz = (self.Vmax - self.Vmin) / float(self.num_atoms - 1)
        self.z = torch.arange(self.Vmin, self.Vmax + self.dz, self.dz)
        
        #Define the model
        if len(state_space.shape) == 3:
        	self.model = model.Model(n_in = state_space.shape, n_out= act_space.n).cuda()
        	self.target_model = model.Model(n_in = state_space.shape, n_out  =act_space.n).cuda()
        	self.state_dtype = torch.uint8
		else:
            print ("observation shape not supported:", obs_space.shape)
            raise
        
        self.model.train()
        self.target_model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9,0.99), eps=1e-8)
		self.replay = Replay.NaiveReplay(replay_size)




    def compute_epsilon(self):
        eps = 1.0 + (0.01 - 1) * self.num_act_steps / self.eps_decay
        eps = max(0.01, eps)
        return eps

    def get_action(self, state):
    	eps = self.compute_epsilon()
        if random.random() <= eps:
            action_idx = random.randrange(self.action_size)
        else:
            action_idx = self.optimal_action(state)
        return action_idx


	def get_optimal_action(self, state):
        dist_list, prob = self.model.forward(state) 
        q_val = np.sum(np.multiply(prob, np.array(self.z)), axis=1) 
        action_idx = np.argmax(q_val)        
        return action_idx

    def train(self, batch_size):
    	[states, actions, rewards, next_states, done] 
    	= self.replay.sample_torch(self.batch_size, self.device)


    	curr_dist, curr_prob  = self.model.forward(next_states)
    	curr_dist_, curr_prob_ = self.target_model.forward(next_states)

    	# Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        for i in range(batch_size):
        	optimal_action_idxs.append(get_action(next_states))

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(batch_size):
            for j in range(self.n_atoms):					
				Tz = reward + (1 - done) * self.discount * self.z[j]
                Tz = torch.clamp(Tz, Vmin, Vmax)
                bj = (Tz - self.Vmin) / self.dz 
                m_l, m_u = torch.floor(bj).long.item(), torch.ceil(bj).long.item()
                m_prob[action[i]][i][int(m_l)] += curr_prob_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += curr_prob_[optimal_action_idxs[i]][i][j] * (bj - m_l)
            
        loss = torch.sum(states * (torch.log(states) - torch.log(m_prob)))


        self.optimizer.zero_grad()
        loss.backward()
		#torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

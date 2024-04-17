import config as cfg
from memory import Memory
from models import Model

import torch
import numpy as np


class Agent:
    def __init__(self, gamma, epsilon, lr,mem_size, batch_Size, num_residuals=cfg.num_residuals,
                 eps_min=0.01, eps_dec=5e-7,replace=1000, chkpt_dir=cfg.chkpt_dir) -> None:
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = 729
        self.input_dims = [9,9]
        self.batch_size = batch_Size
        self.num_residuals = num_residuals
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0

        self.action_space = [i for i in range(self.n_actions)]

        self.memory = Memory(mem_size, self.input_dims)

        self.q_eval = Model(name='model_eval', chkpt_dir=self.chkpt_dir, num_action=self.n_actions, in_channel=1)
        self.q_eval.print_num_parameter()
        
        self.q_next = Model(name='model_next', chkpt_dir=self.chkpt_dir, num_action=self.n_actions, in_channel=1)


    def random_possible_action_space(self,obser):
        l = np.array([], dtype=np.int32)
        for r in range(len(obser[0])):
            for c in range(len(obser[1])):
                if not obser[r][c] ==0:
                    n = np.array([i for i in range(9)], dtype=np.int32)
                    n = n*81
                    n = n + r*9 + c
                    l = np.append(l,n)
        return l

    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(observation, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(self.q_eval.device)
                _,_, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else:
            remove_action = self.random_possible_action_space(observation)
            posi_action = np.delete(self.action_space, remove_action)
            action = np.random.choice(posi_action)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt ==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self,chkpt):
        self.q_eval.save_checkpoint(chkpt)
        self.q_next.save_checkpoint(chkpt)

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).unsqueeze(1).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        states_ = torch.tensor(new_state).unsqueeze(1).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        self.q_eval.train()
        q_pred,_,_ = self.q_eval.forward(states)

        with torch.no_grad():
            q_next,_,_ = self.q_next.forward(states_)

        q_eval,_,_ = self.q_eval.forward(states_)

        max_actions = torch.argmax(q_eval, dim=1)

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        q_target[dones] = False

        self.q_eval.optimizer.zero_grad()
        loss = self.q_eval.loss(q_pred[indices, actions], q_target).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter +=1
        
        self.decrement_epsilon()







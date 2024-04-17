import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config as cfg

from .blocks import Residual

#   dueling DQN
class Model(nn.Module):
    def __init__(self, name, chkpt_dir, num_action,in_channel, hid_channel=cfg.hid_channel, num_residuals=cfg.num_residuals,lr=cfg.lr):
        super(Model, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir,name)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,hid_channel,3,1,1),
                                   nn.BatchNorm2d(hid_channel),
                                   nn.ReLU())
        
        self.residual = nn.ModuleList([Residual(hid_channel,3,1,1,residual=True) for _ in range(num_residuals)])

        self.v_1 = nn.Sequential(nn.Conv2d(hid_channel,1,1,1,0),
                                 nn.BatchNorm2d(1),
                                 nn.ReLU())
        
        self.v_2 = nn.Sequential(nn.Linear(81*1,256),
                                 nn.ReLU(),
                                 nn.Linear(256,1),
                                 nn.Tanh())
                                   
        self.a_1 = nn.Sequential(nn.Conv2d(hid_channel,2,1,1,0),
                                 nn.BatchNorm2d(2),
                                 nn.ReLU())
        
        self.a_2 = nn.Sequential(nn.Linear(81*2,256),
                                 nn.ReLU(),
                                 nn.Linear(256,num_action))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        x = self.conv1(state)
        for f in self.residual:
            x = f(x)

        v = self.v_1(x)
        v = v.view(-1,1*9*9)
        v = self.v_2(v)

        a = self.a_1(x)
        a = a.view(-1,2*9*9)
        a = self.a_2(a)

        q = v + (a - torch.mean(a, dim=1, keepdim=True))

        return q,v,a
        
    def save_checkpoint(self,chkpt):
        print("--   saving checkpoint   --")
        torch.save(self.state_dict(), f"{self.checkpoint_file}_{chkpt}")

    def load_checkpoint(self):
        print("--   loading checkpoint   --")
        self.load_state_dict(torch.load(self.checkpoint_file))

    def print_num_parameter(self):
        return print('total trainable params {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

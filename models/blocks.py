import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """
    residual layer

    input
    
    conv2d
    batchnorm
    relu
    conv2d
    batchnorm

    add input

    relu
    output
    """
    def __init__(self, channel, kernel_size, stride, padding, residual=False):
        super(Residual,self).__init__()
        self.residual_block = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size, stride, padding),
                            nn.BatchNorm2d(channel),
                            nn.ReLU(),
                            nn.Conv2d(channel, channel, kernel_size, stride, padding),
                            nn.BatchNorm2d(channel)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.residual_block(x)
        if self.residual:
            out += x
        return self.act(out)
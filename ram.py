'''
ram.py

Recurrent Attention Model (RAM) Project
PyTorch
Recurrent Attention Model implementation 
'''

import torch
import torch.nn as nn
from torch.distributions import Normal

from network import GlimpseNetwork, CoreNetwork, ActionNetwork, LocationNetwork, Baseline
                                         

# BUILD NETWORK
class RAM(nn.Module):
    def __init__(self, patch_size, 
                       num_scales, 
                       num_channels, 
                       hidden_dim,
                       num_classes, 
                       std):
        super(RAM, self).__init__()
        self.std = std
        self.glimpse = GlimpseNetwork(patch_size, num_scales, num_channels, theta3=hidden_dim)
        self.core = CoreNetwork(hidden_dim, hidden_dim)
        self.locator = LocationNetwork(hidden_dim, std)
        self.classifier = ActionNetwork(hidden_dim, num_classes)
        self.base = Baseline(hidden_dim, 1)
        
  
    def forward(self, img, loc, hidden_state, last=False):
        g = self.glimpse(img, loc)
        h = self.core(g, hidden_state)
        mean, l = self.locator(h)
        b = self.base(h).squeeze() # .squeeze removes dim of h of size 1
        
        log_pi = Normal(mean, self.std).log_prob(l) # pdf of joint is product of pdfs
        log_pi = torch.sum(log_pi, dim=1) # log of product is sum of logs
        
        if ( last ):
            log_prob = self.classifier(hidden_state)
            return h, l, b, log_pi, log_prob
        
        return h, l, b, log_pi
    

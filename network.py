'''
network.py

Implements GlimpseNetwork, CoreNetwork, ActionNetwork,
LocationNetwork, and Baseline class.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from glimpseSensor import GlimpseSensor

class GlimpseNetwork(nn.Module):
    def __init__(self, patch_size, num_scales, channels, theta1=128, theta2=128, theta3=256):
        super(GlimpseNetwork, self).__init__()
        self.glimpseSensor = GlimpseSensor(patch_size, num_scales)
        inpt = channels * patch_size * patch_size * num_scales
        self.fc1 = nn.Linear(inpt, theta1) # takes in glimpses
        self.fc2 = nn.Linear(2, theta2)  # takes in locations
        self.fc3 = nn.Linear(theta1, theta3)
        self.fc4 = nn.Linear(theta2, theta3)
    
    def forward(self, img, loc):
        glimpse = self.glimpseSensor.extract_patch(img, loc)
        x_g = F.relu( self.fc1(glimpse) )
        x_l = F.relu( self.fc2(loc) )
        x = F.relu( self.fc3(x_g) + self.fc4(x_l) )
        return x


class CoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CoreNetwork, self).__init__()
        self.external = nn.Linear(input_dim, hidden_dim)
        self.internal = nn.Linear(hidden_dim, hidden_dim)
      
    
    def forward(self, glimpse_features, internal_state):
        x1 = self.external(glimpse_features) 
        x2 = self.internal(internal_state) 
        x = F.relu( x1 + x2 )
        return x
        
        
class LocationNetwork(nn.Module):
    def __init__(self, input_dim, sd):
        super(LocationNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 2)
        self.sd = sd
        
    def forward(self, hidden_state):
        # .detach() prevents gradient from backpropping through this node
        mean = torch.tanh( self.fc(hidden_state.detach()) ) 
        noise = torch.zeros_like(mean)
        noise.data.normal_(std=self.sd) 
        loc = mean + noise  # add stochastic property to choosing location
        loc = torch.tanh(loc)  # bound between [-1, +1] if noise makes it to big
        return mean, loc
        

class ActionNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActionNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_actions)
        
    def forward(self, hidden_state):
        return F.log_softmax(self.fc(hidden_state), dim=1)
        
        
class Baseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Baseline, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
         
    def forward(self, hidden_state):
        return F.relu( self.fc(hidden_state.detach()) )
        
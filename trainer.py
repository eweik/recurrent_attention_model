'''
trainer.py

Recurrent Attention Model (RAM) Project
PyTorch
'''
from __future__ import print_function 
from __future__ import division
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import os
import numpy as np
import time
import copy
import shutil
import pickle

from utils import AverageMeter
import utils import configure_tensorboard, log_tensorboard
from utils import print_train_stat, print_valid_stat, print_test_set
import utils import plot_glimpse_loc, isPlot
from ram import RAM


# HELPER FUNCTIONS
def get_reward(y, log_class_prob, num_glimpses):
   _, prediction = torch.max(log_class_prob, 1)
   R = (prediction.detach() == y).float()
   R = R.unsqueeze(1).repeat(1, num_glimpses)
   return R    
   
   
def get_loss(y, log_class_prob, log_pi, baselines, rewards):
    # calculate loss for differentiable modules
    #loss_action = F.nll_loss(log_class_prob, y)
    loss_action = F.cross_entropy(log_class_prob, y)
    loss_baseline = F.mse_loss(baselines, rewards)

    # compute reinforce loss
    adjusted_reward = rewards - baselines.detach()
    loss_reinforce = torch.sum( -log_pi * adjusted_reward, dim=1)
    loss_reinforce = torch.mean(loss_reinforce, dim=0)

    # computer loss
    loss = loss_action + loss_baseline + loss_reinforce
    return loss


def get_accuracy(y, log_class_prob):
    _, prediction = torch.max(log_class_prob, 1)
    correct = (prediction == y).float()
    acc = 100 * (correct.sum() / len(y))
    return acc


def get_average(log_class_prob, baselines, log_pi, num_samples):
    # average
    log_class_prob = log_class_prob.view(num_samples, -1, log_class_prob.shape[-1])
    log_class_prob = torch.mean(log_class_prob, dim=0)
    
    baselines = baselines.contiguous().view(num_samples, -1, baselines.shape[-1])
    baselines = torch.mean(baselines, dim=0) 
    
    log_pi = log_pi.contiguous().view(num_samples, -1, log_pi.shape[-1])
    log_pi = torch.mean(log_pi, dim=0)
    return log_class_prob, baselines, log_pi
    

class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        
        # Glimpse Network parameters
        self.patch_size = config.patch_size # size of first glimpse
        self.num_scales = config.num_scales
        
        # Core Network parameters
        self.internal_dim = config.internal_dim
        self.num_glimpses = config.num_glimpses # num glimpses before classification
        
        # Reinforce parameters
        self.std = config.std
        self.num_samples = config.num_samples
        
        # Data parameters
        if config.train:
            self.trainloader, self.validloader = dataloader
            self.num_train = len(self.trainloader.sampler.indices)
            self.num_valid = len(self.validloader.sampler.indices)
        else:
            self.testloader = dataloader
            self.num_test = len(self.testloader.dataset)
        self.batch_size = config.batch_size
        self.classes = 10
        self.channels = 1
        
        # Training parameters
        self.lr = config.learning_rate
        self.momentum = config.momentum
        self.start_epoch = 0
        self.epochs = config.epochs
        self.train_patience = config.train_patience 
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_factor = config.lr_decay_factor
        
        # Miscellaneous parameters
        self.load_best = config.load_best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.use_tensorboard = config.use_tensorboard
        self.resume_training = config.resume_training
        self.plot_freq = config.plot_freq
        self.print_interval = config.print_interval
        self.best_valid_acc = 0.
        self.counter = 0
        self.model_name = 'ram_{}_{}x{}_{}'.format( config.num_glimpses,
                                                    config.patch_size,
                                                    config.patch_size,
                                                    config.num_scales )
        if config.cluttered_translated:
            self.model_name += '_cl_tr'
                  
        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        # configure tensorboard logging
        if self.use_tensorboard:
            configure_tensorboard(self.logs_dir, self.model_name)
        
        
        self.model = RAM(self.patch_size, 
                         self.num_scales, 
                         self.channels, 
                         self.internal_dim,
                         self.classes, 
                         self.std)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, 
        #                           momentum=self.momentum, nesterov=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=self.lr_decay_step, 
                                             gamma=self.lr_decay_factor)


    def reset(self, batch_size):
        internal = torch.zeros(batch_size, self.internal_dim)
        internal = internal.to(self.device)
        location = torch.Tensor(batch_size, 2).uniform_(-1, 1)
        location = location.to(self.device)
        return internal, location
        
        
    def stop_training(self, valid_acc):
        if (valid_acc > self.best_valid_acc):
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.train_patience:
            print("[!] No improvement in a while, stopping training.")
            return True
        return False
    
    
    def check_progress(self, epoch, valid_acc):
        is_best = valid_acc > self.best_valid_acc
        self.best_valid_acc = max(valid_acc, self.best_valid_acc)
        self.save_checkpoint(
            {'epoch': epoch+1,
             'model_state': self.model.state_dict(),
             'optim_state': self.optimizer.state_dict(),
             'sched_state': self.scheduler.state_dict(),
             'best_valid_acc': self.best_valid_acc
            }, is_best
        )
        return
    
    
    def train(self):
        if self.resume_training:
            self.load_checkpoint(best=False)
        
        for epoch in range(self.start_epoch, self.epochs):
            print( '\nEpoch: {}/{}'.format(epoch+1, self.epochs) )
            
            self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate_one_epoch(epoch)
            print_valid_stat(valid_loss, valid_acc, self.num_valid, self.best_valid_acc)
            
            if self.stop_training(valid_acc):
                return
            self.check_progress(epoch, valid_acc)
        
        return
    
    
    def loop_through_glimpses(self, x, loc, internal):
        locs, log_pi, baselines = [], [], []
        for _ in range(self.num_glimpses - 1):
            internal, loc, base, log_p = self.model(x, loc, internal)
            locs.append( loc[0:9] )
            log_pi.append( log_p )
            baselines.append( base )
            
        # last glimpse, get classification (log_class_prob)
        internal, loc, base, log_p, log_class_prob = self.model(x, loc, internal, last=True)
        locs.append( loc[0:9] )
        log_pi.append( log_p )
        baselines.append( base )
        
        # convert to Tensor objects and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)
        return locs, log_pi, baselines, log_class_prob 
    
                    
    def train_one_epoch(self, epoch):
        losses, accs = AverageMeter(), AverageMeter()
        self.scheduler.step()
        self.model.train()
        for i, (x, y) in enumerate(self.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # initialize location vector and internal state
            hid, loc = self.reset(batch_size=self.batch_size) 
            
            imgs = [ x[0:9] ] # save 10 images for later
    
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # loop through glimpses to get prediction
                glmps, log_pi, baselines, log_class_prob = self.loop_through_glimpses(x, loc, hid)

                # get reward, loss, and accuracy
                R = get_reward(y, log_class_prob, self.num_glimpses)
                loss = get_loss(y, log_class_prob, log_pi, baselines, R)
                acc = get_accuracy(y, log_class_prob)
        
                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()
                
                # store for statistics
                losses.update( loss.item(), x.size(0) )
                accs.update( acc.item(), x.size(0) )
            
            # statistics
            if ( i % self.print_interval == 0 ):
                print_train_stat(epoch+1, i+self.print_interval, x, self.num_train, loss, acc)
            if isPlot(epoch, self.plot_freq, i):
                plot_glimpse_loc(imgs, glmps, self.plot_dir, epoch)
            if self.use_tensorboard:
                log_tensorboard(epoch+1, self.trainloader, i, losses, accs)

        return 
    
    
    def validate_one_epoch(self, epoch):
        losses, accs = AverageMeter(), AverageMeter()
        self.model.eval()
        for i, (x, y) in enumerate(self.validloader):
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.repeat(self.num_samples, 1, 1, 1) # duplicate for expectation sampling
            
            # initialize location vector and internal state
            hid, loc = self.reset(batch_size=self.batch_size*self.num_samples)
            
            with torch.no_grad():
                # loop through glimpses to get prediction
                _, log_pi, baselines, log_class_prob = self.loop_through_glimpses(x, loc, hid)
            
                # aggregate for expectation
                log_class_prob, baselines, log_pi = get_average(log_class_prob, baselines, 
                                                                log_pi, self.num_samples)
            
                # get loss and accuracy
                R = get_reward(y, log_class_prob, self.num_glimpses) 
                loss = get_loss(y, log_class_prob, log_pi, baselines, R)
                acc = get_accuracy(y, log_class_prob)
        
                # store for statistics
                losses.update( loss.item(), x.size(0) )
                accs.update( acc.item(), x.size(0) )
                
            
            if self.use_tensorboard:
                log_tensorboard(epoch+1, self.validloader, i, losses, accs)
        
        return losses.avg, accs.avg
            
    
    def test(self):
        correct = 0
        losses = AverageMeter()
        self.load_checkpoint(best=True)
        
        self.model.eval()
        for i, (x, y) in enumerate(self.testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.repeat(self.num_samples, 1, 1, 1) # duplicate for expectation sampling
            
            # initialize location vector and internal state
            hid, loc = self.reset(batch_size=self.batch_size*self.num_samples)
        
            with torch.no_grad():
                # loop through glimpses to get prediction
                _, log_pi, baselines, log_class_prob = self.loop_through_glimpses(x, loc, hid)
        
                # aggregate for expectation
                log_class_prob, baselines, log_pi = get_average(log_class_prob, baselines, 
                                                                log_pi, self.num_samples)
                
                # get reward, loss, and number of correct
                R = get_reward(y, log_class_prob, self.num_glimpses)
                loss = get_loss(y, log_class_prob, log_pi, baselines, R)
                _, prediction = torch.max(log_class_prob, 1)
                correct += prediction.eq( y.data.view_as(prediction) ).sum()
                
                # store for statistics
                losses.update( loss.item(), x.size(0) )
        
        acc = 100. * correct / self.num_test
        print_test_set(losses.avg, correct, acc, self.num_test)
        return losses.avg, acc

            
    def save_checkpoint(self, state, is_best):
        filename = self.model_name + '_ckpt.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + '_model_best.pth'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))        
        return 
       
        
    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + '_ckpt.pth'
        if best:
            filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self.load_variables(filename, ckpt, best)
        return 
        
        
    def load_variables(self, filename, checkpoint, best):
        self.start_epoch = checkpoint['epoch']
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.scheduler.load_state_dict(checkpoint['sched_state'])
        msg = "[*] Loaded {} checkpoint @ epoch {}".format(filename, self.start_epoch)
        if best:
            msg += " with best valid acc of {:.3f}".format(self.best_valid_acc)
        print(msg)
        return
        
        
        
'''
baseline.py

Recurrent Attention Model (RAM) Project
PyTorch
Fully Connected Neural Network Baseline implementation
Convolutional Neural Network Baseline implementation
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time

import dataloader
from utils import AverageMeter
import utils
          

# BUILD NETWORK
class FCNet(nn.Module):
    def __init__(self, img_size):
        super(FCNet, self).__init__()
        self.img_size = img_size
        self.fc1 = nn.Linear(img_size*img_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, self.img_size*self.img_size)
        x = F.relu( self.fc1(x) )
        x = F.dropout(x, training=self.training)
        x = F.relu( self.fc2(x) )
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    def __init__(self, img_size):
        super(ConvNet, self).__init__()
        kernel_size = 10
        stride = 5
        self.len1 = int((img_size - (kernel_size - 1) - 1) / stride ) + 1
        self.conv = nn.Conv2d(1, 8, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(8*self.len1*self.len1, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        #print(x.size())
        x = F.relu( self.conv(x) )
        x = x.view(-1, 8*self.len1*self.len1)
        x = F.relu( self.fc1(x) )
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class BaseTrainer(object):
    def __init__(self, base, train, dataloader, cluttered_translated):
        # Data parameters
        if train:
            self.trainloader, self.validloader = dataloader
            self.num_train = len(self.trainloader.sampler.indices)
            self.num_valid = len(self.validloader.sampler.indices)
        else:
            self.testloader = dataloader
            self.num_test = len(self.testloader.dataset)
        self.batch_size = 20 if train else 1000
        self.classes = 10
        self.channels = 1
        self.img_size = 60 if cluttered_translated else 28
        
        # Training parameters
        self.lr = 0.01
        self.momentum = 0.9
        self.epochs = 200
        self.train_patience = 25
        
        # Miscellaneous parameters
        self.ckpt_dir = './ckpt'
        self.print_interval = 100
        self.best_valid_acc = 0.
        self.counter = 0
        
        # Setup network, loss function, and optimizer
        assert ( base in ('FC', 'Conv') )
        if base == 'FC':
            self.model = FCNet(self.img_size)
            self.model_name = 'FC_cl_tr' if cluttered_translated else 'FC'
        else:
            self.model = ConvNet(self.img_size)
            self.model_name = 'Conv_cl_tr' if cluttered_translated else 'Conv'
            
        self.loss_fn = nn.CrossEntropyLoss()
                         
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, 
                                   momentum=self.momentum, nesterov=True)
        # Decay LR by a factor of 0.1 every 20 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total # of Parameters: {}".format(pytorch_total_params))
        print("# of trainable Parameters: {}".format(pytorch_train_params))
        
    def stop_training(self, valid_acc):
        if (valid_acc > self.best_valid_acc):
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.train_patience:
            print("[!] No improvement in a while, stopping training.")
            return True
        return False
    
    
    def check_progress(self, valid_acc):
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.save_best_checkpoint({'model_state': self.model.state_dict(),
                                      'optim_state': self.optimizer.state_dict(),
                                      'best_valid_acc': self.best_valid_acc})
        return
    
    
    def train(self):
        for epoch in range(self.epochs):
            print( '\nEpoch: {}/{}'.format(epoch+1, self.epochs) )
            
            self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate_one_epoch()
            utils.print_valid_stat(valid_loss, valid_acc, self.num_valid, self.best_valid_acc)
            
            if self.stop_training(valid_acc):
                return
            self.check_progress(valid_acc)
        
        return
    
                    
    def train_one_epoch(self, epoch):
        self.scheduler.step()
        self.model.train()
        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
    
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Forward + get model outputs and loss
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
        
                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()
            
            # statistics
            if ( i % self.print_interval == 0 ):
                utils.print_train_stat(epoch+1, i+self.print_interval, inputs, self.num_train, loss)
        return 
    
    
    def validate_one_epoch(self):
        losses, accs = AverageMeter(), AverageMeter()
        
        self.model.eval()
        for i, (inputs, labels) in enumerate(self.validloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                # Forward + get model outputs and loss
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                # Get accuracy
                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).float()
                acc = 100 * (correct.sum() / len(labels))
        
                # store for statistics
                losses.update( loss.item(), inputs.size(0) )
                accs.update( acc.item(), inputs.size(0) )
        
        return losses.avg, accs.avg
            
    
    def test(self):
        correct = 0
        losses = AverageMeter()
        self.load_best_checkpoint()
        self.model.eval()
        for i, (inputs, labels) in enumerate(self.testloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        
            with torch.no_grad():
                # Forward + get model outputs and loss
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq( labels.data.view_as(preds) ).sum()
                
                # store for statistics
                losses.update( loss.item(), inputs.size(0) )
        
        acc = 100. * correct / self.num_test
        utils.print_test_set(losses.avg, correct, acc, self.num_test)
        return losses.avg, acc
    
    
    def save_best_checkpoint(self, state):
        filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)    
        return 
       
        
    def load_best_checkpoint(self):
        print("[*] Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self.load_variables(filename, ckpt)
        return 
        
        
    def load_variables(self, filename, checkpoint):
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        msg = "[*] Loaded {} checkpoint".format(filename)
        msg += " with best valid acc of {:.3f}".format(self.best_valid_acc)
        print(msg)
        return
        

def setup_dirs(data_dir, ckpt_dir):
    for path in [data_dir, ckpt_dir]:
        if not os.path.exists(path):
            os.makedirs(path)  


def main_fc(basetype, train=True, cluttered_translated=False):
    setup_dirs(data_dir='./data', ckpt_dir='./ckpt')
    
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    
    # get dataloaders
    batch_size = 20 if train else 1000
    data_loader = dataloader.get_dataloader( 
        data_dir='./data', batch_size=batch_size,
        random_seed=1, is_train=train,
        valid_size=0.1, shuffle=train,
        pin_memory=torch.cuda.is_available(), 
        cluttered_translated=cluttered_translated
    )
    
    trainer = BaseTrainer(basetype, train, data_loader, cluttered_translated)
    
    if train:
        trainer.train()
    else:
        trainer.test()
    
    return
    
if __name__ == '__main__':
    start = time.time()
    main_fc(basetype='FC', train=True, cluttered_translated=True)
    utils.print_time(start)
    
    
    
        
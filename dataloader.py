'''
dataloader.py

Recurrent Attention Model (RAM) Project
PyTorch
'''
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from modifiedMNIST import ClutteredTranslatedMNIST


def get_train_valid_indices(dataset, valid_size, random_seed, shuffle):
    num_train = len(dataset)
    indices = list( range(num_train) )
    split = int( valid_size * num_train )
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx = indices[split:]
    valid_idx = indices[:split]
    return train_idx, valid_idx  
    

def get_split_dataloader(idx, dataset, batch_size, pin_memory):
    sampler = SubsetRandomSampler(idx)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=2,
                                             pin_memory=pin_memory)
    return dataloader
    

def get_dataset(data_dir, is_train, cluttered_translated):
    if cluttered_translated:
        return ClutteredTranslatedMNIST(data_dir, 
                                       is_train=is_train,
                                       num_patches=4, 
                                       size_patches=8)
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(root=data_dir, 
                          train=is_train, 
                          download=True, 
                          transform=data_transform)
    

                                    
def get_dataloader(data_dir,
                   batch_size,
                   random_seed,
                   is_train=True,
                   valid_size=0.1,
                   shuffle=True,
                   pin_memory=False,
                   cluttered_translated=False):
    '''
    pin_memory : if True, copy tensors into CUDA pinned memory
    '''
    assert ( (valid_size >= 0) and (valid_size <= 1) )
                                
    dataset = get_dataset(data_dir, is_train, cluttered_translated)
    
    if is_train:
        train_idx, valid_idx = get_train_valid_indices(dataset, 
                                                       valid_size,
                                                       random_seed,
                                                       shuffle)
                                
        trainloader = get_split_dataloader(train_idx, dataset, batch_size, pin_memory)
        validloader = get_split_dataloader(valid_idx, dataset, batch_size, pin_memory)
        return trainloader, validloader
    
    testloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             num_workers=2,
                                             pin_memory=pin_memory)  
    return testloader



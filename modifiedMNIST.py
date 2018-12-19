'''
modifiedMNIST.py

Recurrent Attention Model (RAM) Project
PyTorch

Create cluttered, non-centered images by first placing 
an MNIST digit in a random location of a larger blank
image (60 by 60) and then adding 4 random 8 by 8 subpatches
from other random MNIST digits to random locations of the
image.
'''
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
import numpy as np


def get_random_patches(data_dir, is_train, patch_size):
    data_transform = transforms.Compose([
            transforms.RandomAffine(90),
            transforms.RandomCrop( patch_size ),
            transforms.Pad(26, fill=0),
            transforms.RandomAffine(0, translate=(0.4, 0.4)),
            transforms.ToTensor()
    ])
    
    dataset = datasets.MNIST(root=data_dir, train=is_train, 
                             download=True, transform=data_transform)
    return dataset
        
        
class AddSubpatch(object):
    def __init__(self, data_dir, is_train, num_rand_patches, size_patches):
        self.num_rand_patches = num_rand_patches
        self.size_patches = size_patches
        self.patches = get_random_patches( data_dir, is_train, self.size_patches )
        self.patch_set = [ i[0].squeeze_(0) for i in self.patches ]        
        
    def __call__(self, sample):
        random_patches = np.random.randint(low=0, high=len(self.patch_set)-1,
                                           size=self.num_rand_patches)
        
        img = sample[0]     
        for i in random_patches:
            img.add_( self.patch_set[i] )
            img.add_( sample[0] )
        img.unsqueeze_(0) 
        return img


class ClutteredTranslatedMNIST(Dataset):
    def __init__(self, data_dir, is_train, num_patches, size_patches):
        subpatchAddTransform = AddSubpatch(data_dir, is_train, num_patches, size_patches)
        self.transform = transforms.Compose([
                            transforms.Pad(16),
                            transforms.RandomAffine(0, translate=(0.25, 0.25)),
                            transforms.ToTensor(),
                            subpatchAddTransform,
                            transforms.Normalize((0.56944,), (2.6841,))
                            
                            
        ])
        self.dataset = datasets.MNIST(root=data_dir, train=is_train, 
                                      download=True, transform=self.transform)
                                      
        return
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    

def get_dataloader(data_dir):
    dataset = ClutteredTranslatedMNIST(data_dir, num_patches=4, size_patches=8)
    #dataset = get_random_patches( data_dir, 8 )
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=60000)
    #train = dataloader.__iter__().next()[0]
    #print('Mean: {}'.format(np.mean(train.numpy(), axis=(0, 2, 3))))
    #print('STD: {}'.format(np.std(train.numpy(), axis=(0, 2, 3))))
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=9, 
                                             shuffle=True)
    return dataloader                        
    

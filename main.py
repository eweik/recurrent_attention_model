'''
main.py

Recurrent Attention Model (RAM) Project
PyTorch
Main program to train RAM model on MNIST
'''
import torch
import os

from trainer import Trainer
from config import get_config
from utils import setup_dirs, print_time
from dataloader import get_dataloader


def main(config):
    setup_dirs(config)
    
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    data_loader = get_dataloader( 
        data_dir=config.data_dir, batch_size=config.batch_size,
        random_seed=config.random_seed, is_train=config.train,
        valid_size=config.valid_size, shuffle=config.shuffle,
        pin_memory=torch.cuda.is_available(), 
        cluttered_translated=config.cluttered_translated
    )
    
    trainer = Trainer(config, data_loader)
    
    if config.train:
        trainer.train()
    else:
        trainer.test()
    
    return
    
if __name__ == '__main__':
    config, _ = get_config()
    start = time.time()
    main(config)
    print_time(start)
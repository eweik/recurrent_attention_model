'''
config.py

Recurrent Attention Model (RAM) Project
PyTorch
'''
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')
    
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg
    
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

arg_lists = []
parser = argparse.ArgumentParser(description="Recurrent Attention Model (RAM)")

# Glimpse Network parameters
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=8,
                         help='size of smallest extracted patch')
glimpse_arg.add_argument('--num_scales', type=int, default=1,
                         help='# of scaled patches per glimpse')

# Core Network parameters
core_arg = add_argument_group('Core Network Params')
core_arg.add_argument('--internal_dim', type=int, default=256,
                      help='# of internal nodes in RNN')
core_arg.add_argument('--num_glimpses', type=int, default=7,
                      help='# of glimpses in image')

# Reinforce parameters
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--std', type=float, default=0.15,
                           help='gaussian policy standard deviation')
reinforce_arg.add_argument('--num_samples', type=int, default=10,
                           help='Monte Carlo sampling for valid and test sets')

# Data parameters
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=200,
                      help='# of images in each batch of data')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle train and valid indices')
                 
# Training parameters
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum value')
train_arg.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Learning rate value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--train_patience', type=int, default=25,
                       help='Number of epochs to wait before stopping training')
train_arg.add_argument('--lr_decay_step', type=int, default=25,
                       help='Number of steps before decreasing learning rate')
train_arg.add_argument('--lr_decay_factor', type=float, default=0.1,
                       help='Factor by which to decay learning rate')

# Miscellaneous parameters
misc_arg = add_argument_group('Misc. Params')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--load_best', type=str2bool, default=True,
                      help='Load best model or most recent for training')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory in which Tensorboard logs will be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume_training', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--plot_freq', type=int, default=1,
                      help='How frequently to plot glimpses')
misc_arg.add_argument('--print_interval', type=int, default=100,
                      help='How frequently to print statistics in epoch')
misc_arg.add_argument('--cluttered_translated', type=str2bool, default=False,
                      help='Whether to train on cluttered translated MNIST or plain MNIST')
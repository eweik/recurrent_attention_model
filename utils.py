'''
utils.py

Recurrent Attention Model (RAM) Project
PyTorch
'''
import torch
import time
from tensorboard_logger import configure, log_value
from PIL import Image

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
# TRAINER HELPER FUNCTIONS
def print_time(start):
    time_elapsed = time.time() - start
    print('\nComplete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    return
    

def print_train_stat(epoch, batch_idx, data, trainsize, loss, acc):
    print( 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.0f}%'.format(
            epoch, 
            batch_idx * len(data),
            trainsize,
            100. * batch_idx * len(data) / trainsize, 
            loss.item(),
            acc.item() ) )
    return    
            
            
def print_valid_stat(valid_loss, valid_acc, validsize, best_valid_acc):
    correct = int(valid_acc * validsize / 100)
    print('\nValid set:\n    ', end="")
    end = " [*]\n" if valid_acc > best_valid_acc else "\n"
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          valid_loss, 
          correct, 
          validsize,
          valid_acc), end=end)
    return
    

def print_test_set(test_loss, correct, acc, testsize):
    print('\n[*] Test set:\n    ', end="")
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, 
                                                                  correct, 
                                                                  testsize,
                                                                  acc))
    return
    
    
def isPlot(epoch, plot_freq, itr):
    plot = False
    if (epoch % plot_freq == 0) and (itr==0):
        plot = True
    return plot
    
def configure_tensorboard(logs_dir, model_name):
    tensorboard_dir = logs_dir + model_name
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    configure(tensorboard)

def log_tensorboard(epoch, dataloader, i, losses, accs):
    iteration = epoch * len(dataloader) + i
    log_value('train_loss', losses.avg, iteration)
    log_value('train_acc', accs.avg, iteration)
    return
    
def plot_glimpse_loc(imgs, locs, plot_dir, epoch):
    if torch.cuda.is_available():
        imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
        locs = [l.cpu().data.numpy() for l in locs]
    else:
        imgs = [g.data.numpy().squeeze() for g in imgs]
        locs = [l.data.numpy() for l in locs]
    with open(plot_dir + "g_{}.p".format(epoch+1), "wb") as f:
        pickle.dump(imgs, f)
    with open(plot_dir + "l_{}.p".format(epoch+1), "wb") as f:
        pickle.dump(locs, f)
    return

# MAIN HELPER FUNCTION
def setup_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)  

'''
glimpseSensor.py

Implements GlimpseSensor class.
'''
import torch
import torch.nn.functional as F


# Helper Functions
def getGlimpseSideLengths(patch_size, num_scales):
    length_list = []
    length = patch_size
    for i in range( num_scales ):
        length_list.append( length )
        length = length * 2
    return length_list


def convert_coordinates(L, img_length):
    return ( (L + 1) * img_length ) / 2
    
    
def get_patch_of_size(img, center, size):
    topleft, bottomright = get_coordinates(center, size)
    img, topleft, bottomright = pad_image(img, topleft, bottomright, size)
    patches = []
    for i, (x1, y1), (x2, y2) in zip(range(len(img)), topleft, bottomright):
        patches.append( img[i, :, y1:y2, x1:x2] )
    return torch.stack(patches)
    

def get_coordinates(center, size):
    topleft = center.round().int() - ( size // 2 ) 
    bottomright = topleft + size
    return topleft, bottomright


def pad_image(img, topleft, bottomright, size):
    if ( isPatchNecessary(img, topleft, bottomright) ):
        # pad last 2 dim by half of glimpse size
        p2d = ( 1+size//2, 1+size//2, 1+size//2, 1+size//2 ) 
        topleft += 1 + size//2
        bottomright += 1 + size//2
        img = F.pad(img, p2d, "constant", 0)
    return img, topleft, bottomright
    
    
def isPatchNecessary(img, topleft, bottomright):
    img_length = img.shape[-1]
    for (x1, y1), (x2, y2) in zip(topleft, bottomright):
        if ( x1 < 0 or y1 < 0 or x2 > img_length or y2 > img_length ):
            return True
    return False    


class GlimpseSensor(object):
    def __init__(self, patch_size, num_scales):
        self.glimpse_side_lengths = getGlimpseSideLengths( patch_size, num_scales )
        self.reshape_fn = torch.nn.AdaptiveAvgPool2d( patch_size )
        
        
    def extract_patch(self, img, ctr):
        center = convert_coordinates( ctr, img.shape[-1] )
        patches = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
        for size in self.glimpse_side_lengths:  
            new_patch = get_patch_of_size(img, center, size)
            new_patch = self.resize(new_patch, size)
            patches = torch.cat( (patches, new_patch), dim=1 )
        patches = patches.view( patches.shape[0], -1 ) # flatten
        return patches
    
    
    def resize(self, patches, size):
        if size > self.glimpse_side_lengths[0]:
            return self.reshape_fn( patches )
        return patches
        
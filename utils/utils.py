import torch, random, wandb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms

def fix_randseed(randseed):
    r"""Fix random seed"""
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_wandb(mode, loss, acc, loss_val_ori, acc_val_ori, loss_val_sca, acc_val_sca, best_acc_ori, best_acc_sca):
    if mode == 'orientation':
        wandb.log({'ori/train_loss':loss, 'ori/train_acc': acc, 
        'ori/val_loss': loss_val_ori,  'ori/val_acc': acc_val_ori})        
    elif mode == 'scale':
        wandb.log({'sca/train_loss':loss, 'sca/train_acc': acc,
        'sca/val_loss': loss_val_sca, 'sca/val_acc': acc_val_sca})   
    wandb.log({'ori/best_val_acc': best_acc_ori, 'sca/best_val_acc': best_acc_sca}) 


########### ==============================
class Normalize:
    def __init__(self):
       self.normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

    def __call__(self, image):
        return self.normalize(image).unsqueeze(0)
            
class UnNormalize:
    r"""Image unnormalization"""
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        img = image.clone()
        for im_channel, mean, std in zip(img, self.mean, self.std):
            im_channel.mul_(std).add_(mean)
        return img

def normalize_gray_image(image):
    ## 1. resize the input 32x32
    image = F.interpolate(image, size=(32,32), mode='bilinear')
    B, C, H, W = image.shape
    ## 2. compute per-patch normalization
    mean = torch.mean(image.view(B, -1), dim=1)
    std = torch.std(image.view(B, -1), dim=1)

    image = image.view(B, -1).sub( mean.unsqueeze(1) ).div(std.unsqueeze(1))
    image = image.reshape(B, C, H, W )

    return image

def degree_to_radian(degree):
    return degree * np.pi / 180.

def radian_to_degree(radian):
    return radian * 180. / np.pi

def shift_vector_by_gt(vectors, gt_shifts):
    shifted_vector = []
    for vector, shift in zip(vectors, gt_shifts):
        shifted_vector.append(torch.roll(vector, int(shift)))  ## NOTICE : torch.roll shift values to RIGHT side.
    return torch.stack(shifted_vector)



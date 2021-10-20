import os, sys, torch, argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader

import model.loss as loss
from model.model import PatchPoseNet

from datasets.patchpose import PatchPoseScaleOfflineDataset, PatchPoseOrientationOfflineDataset

from utils.utils import fix_randseed, log_wandb
from utils.train_utils import train
from config import get_train_config

## 1. hyperparameters
args = get_train_config()

fix_randseed(randseed=0)

## 2. Configurations
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
output_ori = args.output_ori
output_sca = args.output_sca
softmax_temperature = args.softmax_temperature
dataset_type = args.dataset_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cur_datetime = datetime.now().__format__('_%m%d_%H%M%S')

## 3. Load dataset
dataset_path = 'datasets/patchPose/'

if args.branch == 'sca':
    mode = 'scale'  
    sca_dataset_train = PatchPoseScaleOfflineDataset(dataset_path, output_sca, dataset_type=args.dataset_type, mode='train')
    sca_dataloader_train = DataLoader(sca_dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    dataloader_train = sca_dataloader_train
    
if args.branch == 'ori':
    mode = 'orientation'
    ori_dataset_train = PatchPoseOrientationOfflineDataset(dataset_path, output_ori, dataset_type=args.dataset_type, mode='train')
    ori_dataloader_train = DataLoader(ori_dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    dataloader_train = ori_dataloader_train

sca_dataset_val = PatchPoseScaleOfflineDataset(dataset_path, output_sca, dataset_type="ppa_ppb", mode='val')
sca_dataloader_val = DataLoader(sca_dataset_val, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True)
ori_dataset_val = PatchPoseOrientationOfflineDataset(dataset_path, output_ori, dataset_type="ppa_ppb", mode='val')
ori_dataloader_val = DataLoader(ori_dataset_val, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True)

## 4. Network initialization
net = PatchPoseNet(backbone=args.backbone, output_ori = args.output_ori, output_sca = args.output_sca, use_pretrain=False).cuda()
net.zero_grad()

if args.load:
    net.load_state_dict(torch.load(args.load))

## 5. Loss function & optimizer
criterion = loss.cross_entropy_symmetric
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=args.momentum)

## 6. init distribution normalizers
softmax = nn.Softmax(dim=1)
normalizer = softmax  

## 7.  Logger 
os.makedirs('logs', exist_ok=True)
logpath = os.path.join('logs', cur_datetime )

logpath += '_' + '_ep' + str(epochs) + '_' + args.backbone + '_lr' + str(round(learning_rate, 3)) \
  + '_bsz' + str(args.batch_size)  + '_data' + str(args.dataset_type)  \
  + '_ori' + str(output_ori) + '_sca' + str(output_sca) + '_branch' + args.branch 
os.mkdir(logpath)

## 8. best model selection & alternating optimization options
best_acc_sca = 0
best_acc_ori = 0
best_acc_both = 0

best_acc_sca_epoch = 0
best_acc_ori_epoch = 0
best_acc_both_epoch = 0

## 9. training
for idx, epoch in enumerate(range(epochs)):
    ## 9-1. Training: single branch training.
    net.train()
    loss, acc = train(epoch, dataloader_train, net, criterion, optimizer, mode, normalizer, softmax_temperature, training=True)    

    ## 9-2. Validate for best model selection.
    with torch.no_grad():
        net.eval()
        loss_val_ori, acc_val_ori = train(epoch, ori_dataloader_val, net, criterion, optimizer, "orientation", normalizer, softmax_temperature, training=False)
        loss_val_sca, acc_val_sca = train(epoch, sca_dataloader_val, net, criterion, optimizer, "scale", normalizer, softmax_temperature, training=False)

    ## 9-3. best model update
    acc_val_both = acc_val_ori  + acc_val_sca 
    if acc_val_ori > best_acc_ori:
        best_acc_ori = acc_val_ori
        best_acc_ori_epoch = idx
        torch.save(net.state_dict(), os.path.join(logpath, 'ori_best_model.pt'))
    if acc_val_sca > best_acc_sca:
        best_acc_sca = acc_val_sca
        best_acc_sca_epoch = idx
        torch.save(net.state_dict(), os.path.join(logpath, 'sca_best_model.pt'))
    if acc_val_both  > best_acc_both:   
        best_acc_both = acc_val_both
        best_acc_both_epoch = idx
        torch.save(net.state_dict(), os.path.join(logpath, 'best_model.pt'))

    print("\n epoch {:d} (trn of {:s})  acc : {:.2f}%, loss  : {:.4f} ".format(epoch, mode[:3], acc, loss) )
    if mode == 'orientation':
        print(" epoch {:d} (val acc) -> ori : {:.2f}%, best (epoch {:d}): {:.2f}%/{:.2f}%".format(epoch, acc_val_ori, best_acc_ori_epoch, best_acc_ori, 100) )
    elif mode == 'scale':
        print(" epoch {:d} (val acc) -> sca : {:.2f}%, best (epoch {:d}): {:.2f}%/{:.2f}%".format(epoch, acc_val_sca, best_acc_sca_epoch, best_acc_sca, 100) )

    ## 9-4. log wandb
    log_wandb(mode, loss, acc, loss_val_ori, acc_val_ori, loss_val_sca, acc_val_sca, best_acc_ori, best_acc_sca)

    
print("Done!")

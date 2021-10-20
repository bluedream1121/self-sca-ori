import os, torch, argparse
import torch.nn as nn
import numpy as np

from model.model import PatchPoseNet

from torch.utils.data import DataLoader
from datasets.test_dataset import test_dataset, get_test_list, get_test_imnames
from utils.test_utils import test, scale_evaluation, orientation_evaluation
from config import get_test_config

## hyperparameters
args = get_test_config()

## Load model and make configurations by directory name parsing.
trained_model_name = args.load
scale_hist_size = int(trained_model_name.split('/')[-2].split('sca')[1].split('_')[0])
scale_hist_size_one_way = (scale_hist_size -1) / 2

orient_hist_size = int(trained_model_name.split('/')[-2].split('ori')[1].split('_')[0])
orient_hist_interval = 360 / orient_hist_size

backbone = 'resnet' + trained_model_name.split('resnet')[1].split('_')[0]

net = PatchPoseNet(backbone=backbone, output_ori = orient_hist_size, output_sca = scale_hist_size, use_pretrain=False).cuda()
net.load_state_dict(torch.load(trained_model_name))
net.eval()

## load test list
test_list, dataset_path = get_test_list(args.dataset_type, args.test_set)

print(' trained_model_name : ', trained_model_name, '\n')
print(" scale hist_size " , scale_hist_size, "\n orientation hist_size " , orient_hist_size, '\n')
print(' total test samples: ',  len(test_list), '\n')

## init dataloader
scale_imnames, orientation_imnames = get_test_imnames(test_list)

dataset_scale = test_dataset(dataset_path, scale_imnames, 'scale')
dataloader_scale = DataLoader(dataset_scale, batch_size=args.batch_size, num_workers=0, shuffle=False)
dataset_orientation = test_dataset(dataset_path, orientation_imnames, 'orientation')
dataloader_orientation = DataLoader(dataset_orientation, batch_size=args.batch_size, num_workers=0, shuffle=False)

## Compute the error.
scale_err = test(net, dataloader_scale, 'scale', scale_hist_size_one_way, None)
orientation_err = test(net, dataloader_orientation, 'orientation', None, orient_hist_interval)

## Compute the results.
res_sca = scale_evaluation(scale_err, scale_hist_size_one_way)
res_ori = orientation_evaluation(orientation_err, orient_hist_size)


import argparse
import pdb
import os
import torch
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torch.nn as nn
from PIL import Image
import tqdm
import shutil
import numpy as np
import random

random.seed(100)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Filtering PatchPoseA and PatchPoseB")
    parser.add_argument("--dataset_path", type=str, default="output", help="path to dataset")
    parser.add_argument("--dataset", type=str, choices=["patchPoseA", "patchPoseB", "hpatchesPoseA", "hpatchesPoseB"], default="patchPoseA", help="dataset to use")
    parser.add_argument("--split_ratio", type=float, default=0.98, help="ratio of images to be filtered")
    args = parser.parse_args()

    args.txt_path = os.path.join(args.dataset_path, args.dataset + ".txt")
    filtered_list = os.path.join(args.dataset_path, f"{args.dataset}ImageList")

    train_txt, val_txt, test_txt = (
        open(os.path.join(filtered_list, "train_acquired.txt"), "r"), 
        open(os.path.join(filtered_list, "val_acquired.txt"), "r"), 
        open(os.path.join(filtered_list, "test_acquired.txt"), "r"), 
    )

    # train_txt_pruned, val_txt_pruned, test_txt_pruned = (
    #     open(os.path.join(filtered_list, "train_pruned.txt"), "w"), 
    #     open(os.path.join(filtered_list, "val_pruned.txt"), "w"), 
    #     open(os.path.join(filtered_list, "test_pruned.txt"), "w"), 
    # )
    a = train_txt.readlines()
    b = val_txt.readlines()
    c = test_txt.readlines()
    print(len(a), len(b), len(c))
    all_list = np.concatenate((a,b,c), axis=0)
    random.shuffle(all_list)

    print(len(all_list) , args.split_ratio)
    train_idx = int(len(all_list) * args.split_ratio)
    val_idx = int(len(all_list) * ((1 - args.split_ratio) /2))
    test_idx = len(all_list)
    train_list = all_list[:train_idx]
    val_list = all_list[train_idx:train_idx+val_idx]
    test_list = all_list[train_idx+val_idx:]
    print(len(all_list), train_idx, val_idx, test_idx)
    print(len(train_list), len(val_list), len(test_list))
    # exit()
    def file_write(path_list, split):
        print("Save as ", os.path.join(filtered_list, split+"_acquired.txt"))
        with open(os.path.join(filtered_list, split+"_acquired.txt"), 'w') as f:
            for i in path_list:
                f.write(i)
    file_write(train_list, 'train')
    file_write(val_list, 'val')
    file_write(test_list, 'test')
    print("Dataset split is Done!")

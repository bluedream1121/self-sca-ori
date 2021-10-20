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

class BatchifiedPatch(Dataset):

    def __init__(self, img_paths):
        self.img_paths = img_paths
    
    def __getitem__(self, idx):
        return tf.ToTensor()(Image.open(self.img_paths[idx]))

    def __len__(self): 
        return len(self.img_paths)

def parse_txt(path, dataset_path):
    with open(path) as fp:
        lines = fp.readlines()
    img_paths = list(map(lambda line: os.path.join(dataset_path, line.split(" ")[1]), lines))
    return img_paths, lines

def batchify_images(img_paths, per_image):
    
    dset = BatchifiedPatch(img_paths)
    dloader = DataLoader(dset, batch_size=per_image)
    return dloader

def filter_images(args):

    img_paths, raw_txt = parse_txt(args.txt_path, args.dataset_path)
    dloader = batchify_images(img_paths, args.per_image)

    filtered_list = os.path.join(args.dataset_path, f"{args.dataset}ImageList")

    if os.path.exists(filtered_list):
        shutil.rmtree(filtered_list)

    os.mkdir(filtered_list)

    std_array = []
   
    model = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2]).cuda()
    # model = resnet18(pretrained=args.use_pretrained).cuda()
    for data in tqdm.tqdm(dloader):
        feats = model(data.cuda())
        sigma = torch.std(feats, dim=1)
        std_array.append(sigma.mean().detach())
        del sigma, feats
    
    std_array = torch.stack(std_array)
    argsort = torch.argsort(std_array)[-int(args.filter_ratio * len(img_paths) / args.per_image):]
    img_idx = torch.flip(argsort, dims=[0])
    img_pruned_idx = torch.argsort(std_array)[:-int(args.filter_ratio * len(img_paths) / args.per_image)]

    assert len(img_pruned_idx) + len(img_idx) == len(std_array)
    
    train_txt, val_txt, test_txt = (
        open(os.path.join(filtered_list, "train_acquired.txt"), "w"), 
        open(os.path.join(filtered_list, "val_acquired.txt"), "w"), 
        open(os.path.join(filtered_list, "test_acquired.txt"), "w"), 
    )

    train_txt_pruned, val_txt_pruned, test_txt_pruned = (
        open(os.path.join(filtered_list, "train_pruned.txt"), "w"), 
        open(os.path.join(filtered_list, "val_pruned.txt"), "w"), 
        open(os.path.join(filtered_list, "test_pruned.txt"), "w"), 
    )

    fp_dict = {
        "train": train_txt,
        "val": val_txt,
        "test": test_txt
    }

    fp_pruned_dict = {
        "train": train_txt_pruned, 
        "val": val_txt_pruned,
        "test": test_txt_pruned,
    }

    print("Saving txt files")

    for idx in tqdm.tqdm(img_idx):
        for l in raw_txt[args.per_image * idx : args.per_image * (idx + 1)]:
            split = l.rstrip("\n").split(" ")[-1]
            fp_dict[split].write(l.split(" ")[1]+ "\n")

    for idx in tqdm.tqdm(img_pruned_idx):
        for l in raw_txt[args.per_image * idx : args.per_image * (idx + 1)]:
            split = l.rstrip("\n").split(" ")[-1]
            fp_pruned_dict[split].write(l.split(" ")[1]+ "\n")

    train_txt.close(); val_txt.close(); test_txt.close()
    train_txt_pruned.close(); val_txt_pruned.close(); test_txt_pruned.close()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Filtering PatchPoseA and PatchPoseB")
    parser.add_argument("--dataset_path", type=str, default="output", help="path to dataset")
    parser.add_argument("--dataset", type=str, choices=["patchPoseA", "patchPoseB", "hpatchesPoseA", "hpatchesPoseB"], default="patchPoseA", help="dataset to use")
    parser.add_argument("--filter_ratio", type=float, default=0.8, help="ratio of images to be filtered")
    parser.add_argument("--use_pretrained", action="store_true", default=False, help="use pretrained ResNet18 for pruning")
    parser.add_argument("--per_image", type=int, default=468, help="number of patches per image")
    args = parser.parse_args()

    args.txt_path = os.path.join(args.dataset_path, args.dataset + ".txt")

    filter_images(args)


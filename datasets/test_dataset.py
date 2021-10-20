import os, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_test_list(dataset_type, test_set):
    test_list_file = []
    if  'ppa' in dataset_type:
        dataset_path = 'datasets/patchPose/'
        test_list_file.append(os.path.join(os.path.join(dataset_path, 'patchPoseAImageList'), test_set + '_acquired.txt'))
    if  'ppb' in dataset_type:
        dataset_path = 'datasets/patchPose/'
        test_list_file.append(os.path.join(os.path.join(dataset_path, 'patchPoseBImageList'), test_set + '_acquired.txt'))
    if dataset_type == 'hpa':
        dataset_path = 'datasets/hpatchesPose/'
        test_list_file.append(os.path.join(os.path.join(dataset_path, 'hpatchesPoseAImageList'), test_set + '_acquired.txt')  )

    test_list = []
    for test_file in test_list_file:
        with open(test_file) as f:
            a = f.readlines()
            test_list.append(a)
    test_list = np.concatenate(test_list, axis=0)
    return test_list, dataset_path

def get_test_imnames(test_list):
    ## Get scale variant image name
    scale_imnames = []
    for img_name in test_list:
        if 'angle000' in img_name:
            scale_imnames.append(img_name)
    ## Get orientation variant image name
    orientation_imnames = []
    for img_name in test_list:
        if 'scale1.0000' in img_name:
            orientation_imnames.append(img_name)
    return scale_imnames, orientation_imnames

class test_parent(Dataset):
    """ Parent class of test dataset."""
    def __init__(self, dataset_path, image_list, evaluation):
        super(test_parent, self).__init__()
        self.dataset_path = dataset_path
        self.image_list = image_list
        self.evaluation = evaluation ## scale or orientation.

        self.to_torch_img = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_list)
    
    def get_image_rsz(self, imname, patch_size= (32,32)):
        img = Image.open(self.dataset_path+ imname.rstrip('\n')).convert('RGB')
        img = img.resize(patch_size)
        img_torch = self.to_torch_img(img)
        
        return img_torch        

    def get_image_crop(self, imname, patch_size = (32,32)):
        ## Open at dataset_path, by (32,32) patch_size.
        img = Image.open(self.dataset_path+ imname.rstrip('\n')).convert('RGB')
        center_point = np.array(img.size) / 2  ## W, H
        start_x = int(center_point[0] - patch_size[0] / 2)
        start_y = int(center_point[1] - patch_size[1] / 2)
        end_x = int(center_point[0] + patch_size[0] / 2)
        end_y = int(center_point[1] + patch_size[1] / 2)

        img = img.crop( (start_x, start_y, end_x, end_y) )
        img_torch = self.to_torch_img(img)

        return img_torch               

class test_dataset(test_parent):
    def __init__(self, dataset_path, image_list, evaluation):
        super(test_dataset, self).__init__(dataset_path, image_list, evaluation)

    def __getitem__(self, idx):
        trg_imname = self.image_list[idx]

        if self.evaluation == 'scale':
            prefix, postfix = trg_imname.split('scale')
            src_imname = prefix + 'scale1.0000'  + postfix[-5:]
            gt_scale_factor = float(trg_imname.split('scale')[1].split('.jpg')[0])
            gt = gt_scale_factor

        if self.evaluation == 'orientation':
            prefix, postfix = trg_imname.split('angle')
            src_imname  = prefix + 'angle000'  + postfix[3:]
            gt_angle = trg_imname.split('angle')[1].split('_scale')[0]
            gt = float(gt_angle)

        ## resize
        src_img = self.get_image_rsz(src_imname, patch_size = (32,32))
        trg_img = self.get_image_rsz(trg_imname, patch_size = (32,32))

        gt = torch.tensor(gt)

        return src_img, trg_img, gt
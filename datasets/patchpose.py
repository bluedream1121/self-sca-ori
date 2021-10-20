import os, cv2, glob, torch
import numpy as np 
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from math import log

def get_angle(im_name):
    angle = im_name.split('angle')[1][:3]
    return angle

def get_scale(im_name):
    scale = im_name.split('scale')[1][:6]
    return scale

class PatchPoseOfflineDataset(Dataset):
    def __init__(self, dataset_path, target_branch, histogram_size=36, dataset_type='ppa', mode='train'):
        super(PatchPoseOfflineDataset, self).__init__()

        training_data_list = dataset_type.split('_')
        image_list_file = []

        if 'ppa' in training_data_list:
            image_list_file.append(os.path.join(os.path.join(dataset_path,'patchPoseAImageList'), mode + '_acquired.txt'))
            if 'all' in training_data_list:
                image_list_file.append(os.path.join(os.path.join(dataset_path,'patchPoseAImageList'), mode + '_pruned.txt'))

        if 'ppb' in training_data_list:
            image_list_file.append(os.path.join(os.path.join(dataset_path,'patchPoseBImageList'), mode + '_acquired.txt'))
            if 'all' in training_data_list:
                image_list_file.append(os.path.join(os.path.join(dataset_path,'patchPoseBImageList'), mode + '_pruned.txt'))
                
        if 'hpa' in training_data_list:
            image_list_file.append(os.path.join(os.path.join(dataset_path,'hpatchesPoseAImageList'), mode + '_acquired.txt') )     
            if 'all' in training_data_list:
                image_list_file.append(os.path.join(os.path.join(dataset_path,'hpatchesPoseAImageList'), mode + '_pruned.txt'))

        if 'hpb' in training_data_list:
            image_list_file.append(os.path.join(os.path.join(dataset_path,'hpatchesPoseBImageList'), mode + '_acquired.txt') )       
            if 'all' in training_data_list:
                image_list_file.append(os.path.join(os.path.join(dataset_path,'hpatchesPoseBImageList'), mode + '_pruned.txt'))

        if len(image_list_file) == 0:
            print(" No such dataset are defined : ", dataset_type)
            exit()
        
        self.image_list = []
        for image_file in image_list_file:
            ## Load the patch image list.
            with open(image_file, 'r') as f:
                image_list_all = f.readlines()

            dataset_type = None
            if 'PoseA' in image_file:
                dataset_type = 'grid' 
            elif 'PoseB' in image_file:
                dataset_type = 'random'

            ## filtering the dataset list by parsing
            if target_branch == 'ori':
                possible_angles = np.linspace(0, 360, histogram_size+1)[:-1]
                for i in image_list_all:
                    prefix1, prefix2, prefix3, prefix4, ori_str, sca_str = i.rstrip('\n').split('_')
                    if round(float(get_scale(sca_str[:-4]))) == 1:
                        if dataset_type == 'random':
                            self.image_list.append(i)  
                        elif dataset_type == 'grid':
                            if float(get_angle(i)) in possible_angles:
                                self.image_list.append(i) 

            if target_branch == 'sca':
                possible_scales = np.round(4 ** np.linspace(-1, 1, histogram_size), 4)
                for i in image_list_all:
                    prefix1, prefix2, prefix3, prefix4, ori_str, sca_str = i.rstrip('\n').split('_')
                    if float(get_angle(ori_str)) == 0:
                        if dataset_type == 'random':
                            self.image_list.append(i)  
                        elif dataset_type == 'grid':
                            if float(get_scale(sca_str[:-4])) in possible_scales:
                                self.image_list.append(i) 

        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        ## pre-processer
        self.to_torch_img = transforms.ToTensor()
        ## for output displacement computation.
        self.histogram_size = histogram_size
        self.bin_size = 360 / histogram_size

        self.mode = mode
        print(" Number of samples ", mode, " of ", target_branch,  len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def get_image_rsz(self, imname, patch_size= (32,32)):
        img = Image.open(imname.rstrip('\n')).convert('RGB')
        img = img.resize(patch_size)
        img_torch = self.to_torch_img(img)
        
        return img_torch        

    def get_image_crop(self, imname, patch_size = (32,32)):
        ## Open at dataset_path, by (32,32) patch_size.
        img = Image.open(imname.rstrip('\n')).convert('RGB')
        center_point = np.array(img.size) / 2  ## W, H
        start_x = int(center_point[0] - patch_size[0] / 2)
        start_y = int(center_point[1] - patch_size[1] / 2)
        end_x = int(center_point[0] + patch_size[0] / 2)
        end_y = int(center_point[1] + patch_size[1] / 2)

        img = img.crop( (start_x, start_y, end_x, end_y) )
        img_torch = self.to_torch_img(img)

        return img_torch               

class PatchPoseOrientationOfflineDataset(PatchPoseOfflineDataset):
    def __init__(self, dataset_path, histogram_size=36, dataset_type='ppa', mode='train'):
        super(PatchPoseOrientationOfflineDataset, self).__init__(dataset_path, 'ori', histogram_size, dataset_type, mode)
    
    def __getitem__(self, idx):
        im_name = self.image_list[idx].rstrip('\n')

        prefix1, prefix2, prefix3, prefix4, ori_str, sca_str = im_name.split('_')
        im_no_rot_name = prefix1 + '_' + prefix2 + '_' + prefix3 + '_' + prefix4 + '_angle000_' + sca_str

        img_path =  os.path.join(self.dataset_path, im_no_rot_name)
        img_rot_path = os.path.join(self.dataset_path, im_name)

        angle = float(get_angle(ori_str))
        angle = angle / self.bin_size ## angle / bin_size == shifting value.

        ## load the images and ground-truth
        patch_size = (32,32)
        img = self.get_image_rsz(img_path, patch_size)
        img_rot = self.get_image_rsz(img_rot_path, patch_size)
        angle = torch.tensor(angle).float()

        return img, img_rot, angle


class PatchPoseScaleOfflineDataset(PatchPoseOfflineDataset):
    def __init__(self, dataset_path, histogram_size=13, dataset_type='ppa', mode='train'):
        super(PatchPoseScaleOfflineDataset, self).__init__(dataset_path, 'sca', histogram_size, dataset_type, mode)    

    def __getitem__(self, idx):
        im_name = self.image_list[idx].rstrip('\n')

        prefix1, prefix2, prefix3, prefix4, ori_str, sca_str = im_name.split('_')
        im_no_sca_name = prefix1 + '_' + prefix2 + '_' + prefix3 + '_' + prefix4 + '_' + ori_str + '_scale1.0000.jpg'

        img_path =  os.path.join(self.dataset_path, im_no_sca_name)
        img_rot_path = os.path.join(self.dataset_path, im_name)

        scale = float(get_scale(sca_str.split('.jpg')[0]))

        ## currently, only consider histogram size 13. 
        one_way = (self.histogram_size -1) / 2
        scale = round(one_way* log(scale, 4) )   ## e,g, round(6 * log(scale, 4) + int(13/2)) == shifing value
   
        ## image preprocessing 
        patch_size = (32,32)
        img = self.get_image_rsz(img_path, patch_size)
        img_rot = self.get_image_rsz(img_rot_path, patch_size)
        scale = torch.tensor(scale).float()

        return img, img_rot, scale


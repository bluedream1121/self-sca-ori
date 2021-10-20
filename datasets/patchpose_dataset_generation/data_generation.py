import argparse
import os
import pdb
import numpy as np
import cv2
import math
import imutils
import tqdm
import pdb
from PIL import Image
import random

def filter_image(img_path, patch_size):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    usable = H > math.ceil(patch_size * 2 * 1.415) * 2 + 3 and W > math.ceil(patch_size * 2 * 1.415) * 2 + 3
    img = None if not usable else img
    return img, H, W

def generation(args): 

    file_list, file_split = [], dict()
    
    if args.dataset in ["patchPoseA", "patchPoseB"]:
        split_list = ["train", "test", "val"]
        files = [os.path.join(args.image_list_path, fname + "_all.txt") for fname in split_list]
        for split, file in zip(split_list, files):
            with open(file) as fp:
                lines = [os.path.join(args.dataset_path, img_path).rstrip("\n") for img_path in fp.readlines()]
            file_list += lines
            for img_path in lines: 
                file_split[img_path] = split


    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    fp_write = open(os.path.join(args.output_dir, f"{args.dataset}.txt"), "w")
    sift = cv2.SIFT_create(nfeatures=1024) 
    

    for (idx, img_path) in tqdm.tqdm(enumerate(file_list), total=len(file_list)):

        if args.dataset == "patchPoseA":
            rotation_list = np.array([10 * i for i in range(36)])
            scale_list = np.array([2 ** ((i - 6) / 3) for i in range(13)])
            fkey = "_".join(img_path.split("/")[-2:])[:-len(".jpg")]

        elif args.dataset == "patchPoseB": 
            rotation_list = np.concatenate([np.array([0]), np.random.choice([i for i in range(360) if i != 0], 35, replace=False)])
            scale_list = np.concatenate([np.array([1.0]), np.random.choice([i for i in range(37501) if i != 7500], 12, replace=False) / (10 ** 4)  + 0.25])
            fkey = "_".join(img_path.split("/")[-2:])[:-len(".jpg")]


        else: 
            raise Exception("Add an appropriate argument to use the dataset!")
            

        assert len(rotation_list) == 36 and len(scale_list) == 13

        img, H, W = filter_image(img_path, args.patch_size)
        if img is None : continue
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        intermediate_patch_size = math.ceil(args.patch_size * 1.415)

        ## 1. SIFT
        kpts = sift.detect(gray, None)

        ## 2. Harris
        # find Harris corners
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        
        kpts_topk = []
        ## 1. SIFT keypoints selection (exception reject)
        for kpt in kpts:
            x = int(kpt.pt[0])
            y = int(kpt.pt[1])
            if x > intermediate_patch_size * 2 and x < (W - intermediate_patch_size * 2 - 1):
                if y > intermediate_patch_size * 2 and y < (H - intermediate_patch_size * 2 - 1):
                    kpts_topk.append( (x,y) ) 

        if len(kpts_topk) > args.num_patches_per_image // 2:
            kpts_topk = random.sample(kpts_topk, args.num_patches_per_image // 2)

        ## 2. Harris corner exception reject 
        corners_topk = []        
        for kpt in corners:
            x = int(kpt[0])
            y = int(kpt[1])
            if x > intermediate_patch_size * 2 and x < (W - intermediate_patch_size * 2 - 1):
                if y > intermediate_patch_size * 2 and y < (H - intermediate_patch_size * 2 - 1):
                    corners_topk.append( (x,y) ) 

        if len(corners_topk) > args.num_patches_per_image // 2:
            corners_topk = random.sample(corners_topk, args.num_patches_per_image // 2)

        kpts_topk.extend(corners_topk)

        ## exception control : If there are no enough valid points.
        if len(kpts_topk) < args.num_patches_per_image:
            num_of_pad_points =  args.num_patches_per_image-len(kpts_topk)
            print("No valid keypoints are detected : ", num_of_pad_points)
            x = W // 2
            y = H // 2
            center = (x, y)
            for itera in range(num_of_pad_points):
                kpts_topk.append(center)

        random.shuffle(kpts_topk)
        print('len(kpts), len(corners), selected_kpts : ' , len(kpts), len(corners), len(kpts_topk))

        for kpt_idx in range(args.num_patches_per_image):
            
            # ( W_rand, H_rand ) = ( 
            #     np.random.randint(intermediate_patch_size * 2, W - intermediate_patch_size * 2 - 1), 
            #     np.random.randint(intermediate_patch_size * 2, H - intermediate_patch_size * 2 - 1) 
            #     )
            W_rand, H_rand = kpts_topk[kpt_idx]

            fname0 = os.path.join(args.dataset, "{}_{}_angle{:03d}_scale{:06.4f}.jpg".format(fkey, kpt_idx, 0, 1.0))
            
            fname_set = set()
            
            for rot in rotation_list:
                for scale in scale_list:

                    fname = os.path.join(args.dataset, "{}_{}_angle{:03d}_scale{:06.4f}.jpg".format(fkey, kpt_idx, rot, scale))

                    img_rescale = imutils.resize(img, width=int(scale * W))
                    img_crop = img_rescale[
                        int(H_rand * scale) - intermediate_patch_size // 2 : int(H_rand * scale) + intermediate_patch_size // 2,
                        int(W_rand * scale) - intermediate_patch_size // 2 : int(W_rand * scale) + intermediate_patch_size // 2
                        ]
                    img_rotate = imutils.rotate(img_crop, rot)
                    img_final = img_rotate[
                        intermediate_patch_size//2 - args.patch_size // 2 : intermediate_patch_size//2 + args.patch_size // 2,
                        intermediate_patch_size//2 - args.patch_size // 2 : intermediate_patch_size//2 + args.patch_size // 2,
                    ]

                    assert img_final.shape[:2] == (args.patch_size, args.patch_size)
                    
                    img_save = Image.fromarray(img_final)   
                    img_save.save(os.path.join(args.output_dir, fname))

                    fp_write.write("{} {} {} {} {}\n".format(fname0, fname, W_rand, H_rand, file_split[img_path]))
                    fname_set.add(fname)

            assert len(fname_set) == 468
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["patchPoseA", "patchPoseB"], default="patchPoseB", help="data type to generate")
    parser.add_argument("--dataset_path", required=True, type=str, help="path to spair directory")
    parser.add_argument("--image_list_path", default="./", type=str, help="path to a directory containing image lists")
    parser.add_argument("--patch_size", default=32, type=int, help="size of patch to extract")

    parser.add_argument("--output_dir", default="./output", type=str, help="path to generate an output directory")
    parser.add_argument("--num_patches_per_image", default=1, type=int, help="number of patches to be extracted per image")
    parser.add_argument("--seed", default=777, type=int, help="default seed to use")

    args = parser.parse_args()
    np.random.seed(args.seed)

    generation(args)
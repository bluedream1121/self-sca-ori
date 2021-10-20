import numpy as np
import math
import cv2
import os.path
import glob
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
patch_size = 32
nfeatures = 500
noutputs = 25
datadir = os.path.normpath(os.path.join(os.path.dirname(__file__), "hpatches-sequences-release"))
save_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "hpatchesPoseA"))
list_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "hpatchesPoseAImageList"))

save_path = '../hpatchesPose/' + save_path
list_path = '../hpatchesPose/' + list_path

random.seed(1121)

def affine_matrix_decompose(h):
    # normalize h
    h /= h[2][2]
    A11, A12, A21, A22 = h[0][0], h[0][1], h[1][0], h[1][1]
    s_x = math.sqrt(A11 ** 2 + A21 ** 2)
    theta = np.arctan2(A21, A11)
    ms_y = A12 * np.cos(theta) + A22 * np.sin(theta)
    if theta == 0:
        s_y = (A22 - ms_y * np.sin(theta)) / np.cos(theta)
    else:
        s_y = (ms_y * np.cos(theta) - A12) / np.sin(theta)
    
    m = ms_y / s_y
    return theta, s_x, s_y, m

def warpPerspectivePoint(src_point, H):
    # normalize h
    H /= H[2][2]
    src_point = np.append(src_point, 1)
    
    dst_point = np.dot(H, src_point)
    dst_point /= dst_point[2]
    
    return dst_point[0:2]

def warpPerspectivePoints(src_points, H):
    # normalize H
    H /= H[2][2]

    ones = np.ones((src_points.shape[0], 1))
    points = np.append(src_points, ones, axis = 1)
    
    warpPoints = np.dot(H, points.T)
    warpPoints = warpPoints.T / warpPoints.T[:, 2][:,None]

    return warpPoints[:,0:2]


def points_sample(points):
    # remove duplicate keypoints
    points = list(set(points))

    # sampling
    num = nfeatures
    if (len(points) < num):
        num = len(points)
    points = random.sample(points, num)

    points = np.array(points)
    return points
    
def detect_SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    points = []
    for point in kp:
        points.append(point.pt)

    return points_sample(points)


def detect_Harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kp = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.cornerHarris(gray,2,3,0.1)

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

    ## for vis
    # res = np.hstack((centroids,corners))
    # res = np.int0(res)
    # img[res[:,1],res[:,0]]=[0,0,255]
    # img[res[:,3],res[:,2]] = [0,255,0]
    # cv2.imshow('dst',img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    kp = corners
    points = []
    for point in kp:
        # points.append((point[0], point[1]))
        points.append((int(point[0]), int(point[1])))            

    return points_sample(points)

def draw_points(img, points, radius, color, thickness, max_num=-1):
    cnt = 0
    for idx in range(points.shape[0]):
        if (max_num != -1) & (cnt >= max_num):
            break
        img = cv2.circle(img.copy(), (int(points[idx][0]),int(points[idx][1])), radius = radius, color=color, thickness=thickness)
        cnt += 1

    return img

def draw_and_save_image_with_points(seq_name, tar_idx, points_list, img_list, name_list):
    prefix = save_path+'/'+seq_name+'_1_'

    for i in range(len(img_list)):
        img_with_points = draw_points(img_list[i], points_list[i], 3, (0, 0, 255), -1)
        cv2.imwrite(prefix+str(tar_idx)+'_' + name_list[i] + '_img.jpg', img_with_points)

def is_inside(img, patch_size, point):
    p = patch_size // 2
    st_x = st_y = p
    height, width, _ = img.shape
    end_x = width - 1 - p
    end_y = height - 1 - p

    if (st_x > point[0]) | (point[0] > end_x) | \
        (st_y > point[1]) | (point[1] > end_y):
        return False
    return True

def filter_points(points_list, img_list, patch_size):
    out_idxs = []
    n, _ = points.shape

    for idx in range(n):
        is_in = True
        for l in range(len(points_list)):
            if (is_inside(img_list[l], patch_size, points_list[l][idx]) == False):
                is_in = False
                break
        if is_in == False:
                out_idxs.append(idx)

    for idx in reversed(out_idxs):
        for l in range(len(points_list)):
            points_list[l] = np.delete(points_list[l], idx, 0)

    return points_list

def generate_ref_tar_patches(points, warp_points, ref_img, tar_img, theta, s_x, s_y):
    tar_list = []
    patches_cnt = 1
    for idx in range(len(points)):
        ref_x, ref_y = points[idx][0], points[idx][1]
        p = patch_size // 2
        ref_crop = ref_img[ref_y - p: ref_y + p, ref_x - p: ref_x + p]
        tar_x, tar_y = warp_points[idx][0], warp_points[idx][1]
        tar_crop = tar_img[tar_y - p: tar_y + p, tar_x - p: tar_x + p]
        
        angle = int((theta * 180 / np.pi) // 10 * 10)
        scale = (s_x + s_y) / 2
        scale = 1.2 ** math.floor(math.log(scale, 1.2))

        prefix = save_path + '/' + seq_name+'_1-'+str(tar_idx)+'_'+str(patches_cnt)
        ref_crop_name = prefix + '_angle000_scale1.0000.jpg'
        tar_crop_name = prefix + '_angle' + f"{angle:03d}" + '_scale' + f"{scale:.4f}" +'.jpg'

        patches_cnt += 1

        # if (angle > 90) | (angle < -90) :
        #     continue

        # if (scale < (1.2) ** (-4)) | (scale > (1.2) ** 4):
        #     continue


        if ref_crop.size == 0:
            print('ref_crop {} is empty center {}x{}, image {}x{}'.format(ref_crop_name, ref_x, ref_y, ref_img.shape[1], ref_img.shape[0]))
        if tar_crop.size == 0:
            print('tar_crop {} is empty center {}x{}, image {}x{}'.format(tar_crop_name, tar_x, tar_y, tar_img.shape[1], tar_img.shape[0]))
        cv2.imwrite(ref_crop_name, ref_crop)
        cv2.imwrite(tar_crop_name, tar_crop)

        # save image list
        tar_list.append(tar_crop_name)

    return tar_list

def generate_ref_tar_approx_patches(points, warp_points, aff_points, ref_img, tar_img, aff_img, theta, s_x, s_y):
    patches_cnt = 1
    for idx in tqdm(range(len(points)), total=len(points)):
        ref_x, ref_y = points[idx][0], points[idx][1]
        p = patch_size // 2
        ref_crop = ref_img[ref_y - p: ref_y + p, ref_x - p: ref_x + p]
        tar_x, tar_y = warp_points[idx][0], warp_points[idx][1]
        tar_crop = tar_img[tar_y - p: tar_y + p, tar_x - p: tar_x + p]
        aff_x, aff_y = aff_points[idx][0], aff_points[idx][1]
        aff_crop = aff_img[aff_y - p: aff_y + p, aff_x - p: aff_x + p]

        
        angle = (theta * 180 / np.pi)
        scale = (s_x + s_y) / 2
        scale = 1.2 ** math.floor(math.log(scale, 1.2))

        prefix = save_path + '/' + seq_name+'_1-'+str(tar_idx)+'_'+str(patches_cnt)+f"({angle:.2f}, {s_x:.4f}, {s_y:.4f})"
        ref_crop_name = prefix + '_ref.jpg'
        tar_crop_name = prefix + '_tar.jpg'
        aff_crop_name = prefix + '_aff.jpg'

        patches_cnt += 1

        # if (angle > 90) | (angle < -90) :
        #     continue

        # if (scale < (1.2) ** (-4)) | (scale > (1.2) ** 4):
        #     continue

        if ref_crop.size == 0:
            print('ref_crop {} is empty center {}x{}, image {}x{}'.format(ref_crop_name, ref_x, ref_y, ref_img.shape[1], ref_img.shape[0]))
        if tar_crop.size == 0:
            print('tar_crop {} is empty center {}x{}, image {}x{}'.format(tar_crop_name, tar_x, tar_y, tar_img.shape[1], tar_img.shape[0]))
        if aff_crop.size == 0:
            print('aff_crop {} is empty center {}x{}, image {}x{}'.format(aff_crop_name, aff_x, aff_y, aff_img.shape[1], aff_img.shape[0]))
        # cv2.imwrite(ref_crop_name, ref_crop)
        # cv2.imwrite(tar_crop_name, tar_crop)
        # cv2.imwrite(aff_crop_name, aff_crop)

        fig = plt.figure()
        rows = 1
        cols = 3
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        
        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(cv2.cvtColor(tar_crop, cv2.COLOR_BGR2RGB))
        ax2.axis("off")
        
        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(cv2.cvtColor(aff_crop, cv2.COLOR_BGR2RGB))
        ax3.axis("off")
        save_dir ='vis'

        save_name = ref_crop_name[:-8] + '.jpg'
        plt.savefig(os.path.join(save_dir, save_name))
        plt.clf()
        plt.close()
        break

def dataset_split(file_list):    
    type_ratio = {
        'train_acquired':8,
        'val_acquired':1,
        'test_acquired':1
    }

    idx = 0
    data_dic = {}
    ratio_list = list(type_ratio.values())
    ratio_total = sum(ratio_list)
    for type, ratio in type_ratio.items():
        cnt = int(len(file_list) * (ratio / ratio_total))
        data_dic[type] = file_list[idx:idx+cnt]
        idx += cnt
    return data_dic


os.makedirs(save_path, exist_ok=True)
os.makedirs(list_path, exist_ok=True)

total_pair = 0
selected_pair = 0 

tar_paths = []
statistics = []
seqs = glob.glob(datadir+'/v_*')
for seq in tqdm(seqs, total=len(seqs)):
    seq_name = seq.split('/')[-1]

    ref_idx = 1
    ref_path = seq+'/'+str(ref_idx)+'.png'
    if os.path.exists(ref_path) == False:
        continue
    
    # load reference image
    ref_img = cv2.imread(ref_path)

    # detect SIFT keypoints
    points1 = detect_SIFT(ref_img)
    points2 = detect_Harris(ref_img)

    # print(points1.shape, points2.shape)
    points = np.concatenate((points1, points2), axis=0)
    # print(points.shape)
    for tar_idx in range(2, 7):
        H_path = seq+'/H_1_'+str(tar_idx)
        tar_path = seq+'/'+str(tar_idx)+'.png'
        if (os.path.exists(H_path) == False) | \
            (os.path.exists(tar_path) == False):
            continue
        
        total_pair += 1
        # load target image
        tar_img = cv2.imread(tar_path)

        # load homography matrix
        H = np.fromfile(H_path, sep=" ")
        H.resize((3, 3))

        theta, s_x, s_y, m = affine_matrix_decompose(H)
        statistics.append([theta * 180 / np.pi, s_x, s_y])

        ## this is shearing remove
        # if abs(m) > 0.1:
        #     continue

        selected_pair += 1

        # warp keypoint with homography matrix
        warp_points = warpPerspectivePoints(points, H)

        # # # # warp keypoint with approximated affine matrix
        # visualize = True
        # if visualize:
        #     A = np.zeros((3,3))
        #     A[0][0] = s_x * np.cos(theta)
        #     A[0][1] = -s_y * np.sin(theta)
        #     A[1][0] = s_x * np.sin(theta)
        #     A[1][1] = s_y * np.cos(theta)
        #     A[2][2] = 1
        #     aff_points = warpPerspectivePoints(points, A)
        #     aff_img = cv2.warpPerspective(ref_img, A, (ref_img.shape[1], ref_img.shape[0]))
            
        #     points = np.around(points).astype(np.int)
        #     warp_points = np.around(warp_points).astype(np.int)
        #     aff_points = np.around(aff_points).astype(np.int)
        #     points, warp_points, aff_points = filter_points([points, warp_points, aff_points], [ref_img, tar_img, aff_img], patch_size)
            
        #     print(points.shape, warp_points.shape, aff_points.shape)
        #     if points.shape[0] < noutputs:
        #         continue
        #     ## sample 25 points.
        #     assert points.shape == warp_points.shape == aff_points.shape
        #     select = random.sample(list(range(points.shape[0])), noutputs)
        #     points_select = points[select]
        #     warp_points_select = warp_points[select]
        #     aff_points_select = aff_points[select]
        #     assert points_select.shape == warp_points_select.shape ==aff_points_select.shape

        #     print(points_select.shape, warp_points_select.shape, aff_points_select.shape)
        #     draw_and_save_image_with_points(seq_name, tar_idx, [points_select, warp_points_select, aff_points_select], [ref_img, tar_img, aff_img], ['ref', 'tar', 'aff'])

        #     # generate ref, tar, approximation patch pairs.
        #     generate_ref_tar_approx_patches(points_select, warp_points_select, aff_points_select, ref_img, tar_img, aff_img,  theta, s_x, s_y)
        #     print(f"{seq_name}_1_{tar_idx} : {m:.5f}")
        

        # # filter out of bound points
        points = np.around(points).astype(np.int)
        warp_points = np.around(warp_points).astype(np.int)
        points, warp_points = filter_points([points, warp_points], [ref_img, tar_img], patch_size)
        assert points.shape == warp_points.shape 
        select = random.sample(list(range(points.shape[0])), noutputs)
        points_select = points[select]
        warp_points_select = warp_points[select]
        assert points_select.shape == warp_points_select.shape

        # # save reference and target image with points
        draw_and_save_image_with_points(seq_name, tar_idx, [points_select, warp_points_select], [ref_img, tar_img], ['ref', 'tar'])

        # generate patch pairs
        tar_list = generate_ref_tar_patches(points_select, warp_points_select, ref_img, tar_img, theta, s_x, s_y)

        tar_paths.extend(tar_list)

print('total_pair, selected_pair : ', total_pair, selected_pair)
print(len(tar_paths))
# exit()
with open(list_path +'/'+'test_acquired.txt', 'w') as f:
    for item in tar_paths:
        f.write("%s\n" % item)
random.shuffle(tar_paths)
# data_dic = dataset_split(tar_paths)
# for data_type, paths in data_dic.items():
#     with open(list_path + '/' + data_type + '.txt', 'w') as f:
#         for item in paths:
#             f.write("%s\n" % item)
#         f.close()

# with open(save_path+'/statistics.txt', 'w') as file:
#     file.writelines('\t'.join(str(j) for j in i) + '\n' for i in statistics)

        


    

        


        
        

import os, time, math, torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def test(net, dataloader, evaluation, scale_hist_size_one_way, orient_hist_interval):
    err = []
    tic = time.time()
    softmax = nn.Softmax(dim=0)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        src_img, trg_img, gt = batch
        src_img, trg_img, gt = src_img.cuda(), trg_img.cuda(), gt.cuda()

        res1_ori, res1_sca = net(src_img)
        res2_ori, res2_sca = net(trg_img)
        if evaluation == 'scale':
            gt = torch.log2(gt) / 2 * scale_hist_size_one_way  ## math.log(gt, 4) == torch.log2() / 2
            res1 = res1_sca
            res2 = res2_sca
        elif evaluation == 'orientation':
            gt = gt / orient_hist_interval
            res1 = res1_ori
            res2 = res2_ori

        pred1 = torch.argmax(res1, dim=1)
        pred2 = torch.argmax(res2, dim=1)
        res = torch.stack(((pred2 - pred1).float(), gt.float()), dim=0)

        err.append(res)

    toc = time.time()
    print("total " , evaluation, " evaluation time : ", round(toc-tic, 3), " (sec.)  with ", len(dataloader.dataset) ," pairs")

    err = torch.cat([i for i in err], dim=1).t()
    return err

def scale_evaluation(scale_err, scale_hist_size_one_way, scale_thres=[0.5, 1, 1.5, 2]):
    print(" \n Evaluate scale ")
    ## 0.5, 1, 2 -> 2^(1/6), 2^(1/3), 2^(2/3)  (at histogram 13)
    scale_thres = np.array(scale_thres) / int(6 / scale_hist_size_one_way) ## difference of bin idx. (at histogram 13)
    res_scale = []
    scale_res = {}
    scale_diff = []
    for thres in scale_thres:
        scale_res[thres] = [] 
        for sc in scale_err:
            a = float(sc[0])
            b = round(float(sc[1]))

            diff = torch.abs(torch.tensor([a - b]))
            scale_diff.append(diff / 3)
            if diff <= thres:
                scale_res[thres].append(True)
            else:
                scale_res[thres].append(False)
        
        print("accuracy at threshold ", round(2 ** (thres/3),4) , "(", thres, ")" , " : ", round(np.sum(scale_res[thres])/ len(scale_res[thres]), 4))
        res_scale.append(round(np.sum(scale_res[thres]) *100 / len(scale_res[thres]), 2) )

    print(res_scale)

    return res_scale

def orientation_evaluation(orientation_err, orient_hist_size, ori_thres=[0.5,1,2,4]):
    print(" \n Evaluate orientation ")
    ## 0.5, 1, 2 -> 5 degrees, 10 degrees, 20 degrees (at histogram 36)
    orientation_thres = np.array(ori_thres) / int(36 / orient_hist_size )
    res_ori = []
    orientation_res = {}
    orientation_diff = []
    for thres in orientation_thres:
        orientation_res[thres] = [] 
        for ori in orientation_err:
            diff = torch.min ( torch.tensor([torch.abs(ori[0] - ori[1]) , \
                                torch.abs(ori[0] - ori[1] - orient_hist_size) , \
                                torch.abs(ori[0] - ori[1] + orient_hist_size )] ) )  ## orientation is cyclic.

            orientation_diff.append(diff * (360 / orient_hist_size))

            if diff <= thres: 
                orientation_res[thres].append(True)
            else:
                orientation_res[thres].append(False)
        print("accuracy at threshold ", thres * int(360 / orient_hist_size) , "(", thres, ")"  , " : ", round(np.sum(orientation_res[thres])/ len(orientation_res[thres]), 4))
        res_ori.append(round(np.sum(orientation_res[thres]) * 100/ len(orientation_res[thres]), 2) )

    print(res_ori)

    return res_ori
import torch, math
from tqdm import tqdm

def train(epoch, dataloader, net, criterion, optimizer, mode, normalizer, softmax_t, training):
    loss_sum = 0
    accuracy = 0
    accuracy_prev = 0
    temperature = float(softmax_t)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        img, img_aug, gt = batch  ## img_aug : img_rot or img_scale. gt is shifting factor.
        img, img_aug, gt = img.cuda(), img_aug.cuda(), gt.cuda()

        res_ori, res_sca = net(img)
        res_ori_aug, res_sca_aug = net(img_aug)

        if mode == 'orientation':
            loss, _, correct_vector = train_orientation(res_ori, gt, normalizer, res_ori_aug, criterion, temperature)
        elif mode == 'scale':
            loss, _, correct_vector = train_scale(res_sca, gt, normalizer, res_sca_aug, criterion, temperature)

        loss_sum += loss
        accuracy += torch.sum(correct_vector)

        if training:
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()   

    return loss_sum / len(dataloader.dataset), float(int(accuracy) / len(dataloader.dataset)) *100


def train_scale(res_sca, gt, normalizer, res_sca_aug, criterion, temperature):
    res_shift = shift_vector_by_gt_float(res_sca, gt)
    res_shift = scale_shift_vector_modification(res_shift, gt.long())  ## scale gt can be negative integer.
    
    loss = criterion(normalizer(res_sca_aug / temperature ), normalizer(res_shift / temperature ))
    loss = torch.sum(loss) / gt.shape[0]  
    ## for evaluation
    pred_scale_diff =( torch.argmax(res_sca_aug, dim=1) - torch.argmax(res_sca, dim=1)) 
    correct_vector = (pred_scale_diff == gt)

    return loss, pred_scale_diff, correct_vector

def train_orientation(res_ori, gt, normalizer, res_ori_aug, criterion, temperature):
    res_shift = shift_vector_by_gt_float(res_ori, gt)

    loss = criterion(normalizer(res_ori_aug / temperature ), normalizer(res_shift / temperature ))
    loss = torch.sum(loss) / gt.shape[0] 
    ## for evaluation
    pred_angle_idx =( torch.argmax(res_ori_aug, dim=1) - torch.argmax(res_ori, dim=1)) % res_ori.shape[1] 
    correct_vector = (pred_angle_idx == gt)   

    return loss, pred_angle_idx, correct_vector


def shift_vector_by_gt_float(vectors, gt_shifts):
    shifted_vector = []
    for vector, shift in zip(vectors, gt_shifts):
        shift_floor = math.floor(shift)
        shift_ceil = math.ceil(shift)
        
        if shift_floor == shift_ceil:
            shift_float = torch.roll(vector, int(shift))
        else:
            shift_float = (shift_ceil - shift) * torch.roll(vector, shift_floor) + \
                        (shift - shift_floor) * torch.roll(vector, shift_ceil)
        shifted_vector.append(shift_float)  ## NOTICE : torch.roll shift values to RIGHT side.
    return torch.stack(shifted_vector)

def scale_shift_vector_modification(shifted_vector, gt):
    ## The shifted vector should be probility distribution. (it means all non-zero positive values.)
    modified_shift_vector = []
    for vector, move in zip(shifted_vector, gt):
        if move >= 0:
            mask = torch.cat((torch.zeros(move), torch.ones(len(vector) - move)))
        else:
            mask = torch.cat((torch.ones(len(vector) + move), torch.zeros(-move)))
        vector = vector * mask.cuda()
        modified_shift_vector.append(vector)
    modified_shift_vector = torch.stack(modified_shift_vector)
    return modified_shift_vector
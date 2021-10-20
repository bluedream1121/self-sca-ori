import torch
import torch.nn.functional as F


def kl_div_temperature(pred_p, pred_q, T):
    p = F.log_softmax(pred_p/T, dim=1)
    q = F.softmax(pred_q/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / pred_p.shape[0]
    return l_kl 

def kl_divergence(p, q):
    return - p * (torch.log(q + 1e-5)) + p * (torch.log(p + 1e-5))

def js_divergence(p_gt, q_model_output):
    M = (p_gt + q_model_output) * 0.5
    return (kl_divergence(p_gt, M) + kl_divergence(q_model_output, M)) * 0.5

def cross_entropy_symmetric(p_gt, q_model_output):
    return - p_gt * (torch.log(q_model_output + 1e-5)) - q_model_output * (torch.log(p_gt + 1e-5))

def cross_entropy(p_gt, q_model_output):
    return - p_gt * (torch.log(q_model_output + 1e-5))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import OrderedDict

class PatchPoseNet(nn.Module):
    def __init__(self, backbone, output_ori, output_sca, use_pretrain):
        super(PatchPoseNet, self).__init__()

        final_output_channel = None
        if 'resnet' in backbone: 
            if backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=use_pretrain)
                final_output_channel = 2048
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=use_pretrain)
                final_output_channel = 2048
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(pretrained=use_pretrain)
                final_output_channel = 512
            elif backbone == 'resnet18':
                self.backbone = models.resnet18(pretrained=use_pretrain)            
                final_output_channel = 512
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
            self.orientation_learners = nn.Sequential(  ## classifier
                nn.Linear(final_output_channel, final_output_channel),
                nn.BatchNorm1d(final_output_channel),
                nn.ReLU(),
                nn.Linear(final_output_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_ori)
            )
            self.scale_learners = nn.Sequential(  ## classifier
                nn.Linear(final_output_channel, final_output_channel),
                nn.BatchNorm1d(final_output_channel),
                nn.ReLU(),
                nn.Linear(final_output_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_sca)
            )

        self.GlobalMaxPooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x_ori = self.orientation_learners(self.GlobalMaxPooling(x).view(B, -1))
        x_sca = self.scale_learners(self.GlobalMaxPooling(x).view(B, -1))

        return x_ori, x_sca

    def state_dict(self):
        res = OrderedDict()
        res['backbone'] = self.backbone.state_dict()
        res['orientation_learners'] = self.orientation_learners.state_dict()
        res['scale_learners'] = self.scale_learners.state_dict()
        
        return res

    def load_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict['backbone'])
        self.orientation_learners.load_state_dict(state_dict['orientation_learners'])
        self.scale_learners.load_state_dict(state_dict['scale_learners'])
    
    def eval(self):
        self.backbone.eval()
        self.orientation_learners.eval()
        self.scale_learners.eval()

    def train(self):
        self.backbone.train()
        self.orientation_learners.train()
        self.scale_learners.train()

if __name__ == "__main__":
    net = PatchPoseNet('resnet18',18,13,False).cuda()
    temp = torch.randn(4,3,32,32).cuda() 
    res_ori, res_sc = net(temp)
    print(res_ori.shape, res_sc.shape, temp.shape)

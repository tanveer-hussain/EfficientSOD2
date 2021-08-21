from ResNet import *
import torchvision.models as models
from torch import nn
import torchvision
from deformable_conv import DeformConv2d
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dpt.models import DPTSegmentationModel as DPTModel

class SalModel(nn.Module):
    def __init__(self):
        super(SalModel, self).__init__()


        pself.reprocess_layer_7 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=(3, 3), stride=1,
                                       padding=1).cuda().half()
        self.preprocess_layer_6 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(3, 3), stride=1,
                                       padding=1).cuda().half()

        self.model = DPTModel()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = self.model.half()

        self.LinearNet = nn.Sequential(
           nn.Conv2d(in_channels=150, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
           nn.Conv2d(in_channels=64, out_channels=28, kernel_size=(3, 3), stride=1, padding=1),
           nn.AdaptiveAvgPool2d((21, 21))
        ).cuda().half()

        self.fc1 = nn.Linear(28 * 21 * 21, latent_size).cuda().half()
        self.fc2 = nn.Linear(28 * 21 * 21, latent_size).cuda().half()

    def forward(self,images, depths, gts):
        if 

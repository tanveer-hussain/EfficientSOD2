import torch
from  ResNet_models_Custom import Saliency_feat_encoder, Encoder_x, Encoder_xy
import torch.nn as nn
from ResNet_models_UCNet import Generator
device = torch.device('cuda' if torch.cuda.is_available else "cpud")

class ResSwin():
    def __init__(self, channel, latent_dim):
        self.relu = nn.ReLU(inplace=True)
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        self.xy_encoder = Encoder_xy(7, channel, latent_dim)
        self.x_encoder = Encoder_x(6, channel, latent_dim)

x = torch.randn((8, 3, 224, 224)).to(device)
depth = torch.randn((8, 3, 224, 224)).to(device)
gt = torch.randn((8, 1, 224, 224)).to(device)
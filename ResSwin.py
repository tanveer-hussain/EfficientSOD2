import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder
from DenseSwin import SwinSaliency

class ResSwinModel(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwinModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.swin_saliency = SwinSaliency()
        self.conv1 = nn.Conv2d(3, 1, 3, 1, 1)

        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)

    def forward(self, x, depth, y, training=True):
        if training:
            # self.x_sal, self.d_sal = self.sal_encoder(x, depth)
            self.x_f1 = self.swin_saliency(x)
            self.x_f2 = self.swin_saliency(x)
            self.x_f = torch.cat((self.x_f1, self.x_f2),1)
            print (self.x_f.shape)
            # self.d_sal = self.conv1(self.d_sal)

            return self.x_sal#, self.d_sal #self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            # _, mux, logvarx = self.x_encoder(torch.cat((x, depth), 1))
            # z_noise = self.reparametrize(mux, logvarx)
            # self.prob_pred, _ = self.sal_encoder(x, depth, z_noise)
            # self.prob_pred, _ = self.sal_encoder(x, depth)
            return 0

# x = torch.randn((12, 3, 224, 224)).to(device)
# depth = torch.randn((12, 3, 224, 224)).to(device)
# gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwin(32,3).to(device)
# a, b, c, d, e = model(x,depth, gt)
# print ('done')
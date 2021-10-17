import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder


class ResSwinModel(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwinModel, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        # self.swin_saliency = SwinSaliency()
        # self.conv1 = nn.Conv2d(3, 1, 3, 1, 1)
        # self.liner1024 = nn.Linear(2048, 1024)
        # self.upsampling = nn.Sequential(
        #     nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        # )
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)

    def forward(self, x, training=True):
        if training:
            self.x_sal = self.sal_encoder(x)
            # self.x_sal, self.d_sal = self.sal_encoder(x, depth)
            # print (self.x_sal.shape)
            # self.x_sal , self.d_sal = self.swin_saliency(x, depth)
            # print(self.x_f1.shape)
            # self.x_f2 = self.swin_saliency(x)
            # self.x_f = torch.cat((self.x_f1, self.x_f2),1)
            # self.x_f = self.liner1024(self.x_f)
            # self.x_f = self.x_f1.view(1,32,32)
            # self.x_f = torch.unsqueeze(self.x_f, 0)
            # self.x_sal = self.upsampling(self.x_f)
            # print (self.x_sal.shape)
            # self.d_sal = self.conv1(self.d_sal)

            return self.x_sal #, self.d_sal #self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            self.x_sal = self.sal_encoder(x)
            # self.x_sal, _ = self.sal_encoder(x, depth)
            # _, mux, logvarx = self.x_encoder(torch.cat((x, depth), 1))
            # z_noise = self.reparametrize(mux, logvarx)
            # self.prob_pred, _ = self.sal_encoder(x, depth, z_noise)
            # self.prob_pred, _ = self.sal_encoder(x, depth)
            return self.x_sal

# x = torch.randn((12, 3, 224, 224)).to(device)
# depth = torch.randn((12, 3, 224, 224)).to(device)
# gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwin(32,3).to(device)
# a, b, c, d, e = model(x,depth, gt)
# print ('done')
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder

class ResSwinModel(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwinModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)

    def forward(self, x, depth, y=None, training=True):
        if training:
            # self.x_sal, self.d_sal = self.sal_encoder(x, depth)

            return self.x_sal, self.d_sal #self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            # _, mux, logvarx = self.x_encoder(torch.cat((x, depth), 1))
            # z_noise = self.reparametrize(mux, logvarx)
            # self.prob_pred, _ = self.sal_encoder(x, depth, z_noise)
            self.prob_pred, _ = self.sal_encoder(x, depth)
            return self.prob_pred

# x = torch.randn((12, 3, 224, 224)).to(device)
# depth = torch.randn((12, 3, 224, 224)).to(device)
# gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwin(32,3).to(device)
# a, b, c, d, e = model(x,depth, gt)
# print ('done')
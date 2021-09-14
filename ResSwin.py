import torch
from torch.autograd import Variable
from  ResNet_models_Custom import Saliency_feat_encoder, Encoder_x, Encoder_xy, Triple_Conv
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from torch.distributions import Normal, Independent, kl

from utils import l2_regularisation
import torch.nn.functional as F

class ResSwinModel(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwinModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        self.xy_encoder = Encoder_xy(7, channel, latent_dim)
        self.x_encoder = Encoder_x(6, channel, latent_dim)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, depth, y=None, training=True):
        if training:
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x,depth,y),1))
            self.prior, mux, logvarx = self.x_encoder(torch.cat((x,depth),1))
            lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)
            self.prob_pred_post, self.depth_pred_post  = self.sal_encoder(x,depth,z_noise_post)
            self.prob_pred_prior, self.depth_pred_prior = self.sal_encoder(x, depth, z_noise_prior)
            self.reg_loss = l2_regularisation(self.xy_encoder) + \
                        l2_regularisation(self.x_encoder) + l2_regularisation(self.sal_encoder) + l2_regularisation(self.swinmodel)
            #
            # x = F.interpolate(x, size=64)
            # depth = F.interpolate(depth, size=64)
            # self.x_swin_features = self.swinmodel(x)
            # self.d_swin_features = self.swinmodel(depth)
            #
            # self.x_swin_features = self.TrippleConv2(self.TrippleConv1(self.x_swin_features))
            # self.d_swin_features = self.TrippleConv2(self.TrippleConv1(self.d_swin_features))
            #
            # self.x_swin = self.upsample(self.upsample3(self.x_swin_features))
            # self.d_swin = self.upsample(self.upsample3(self.d_swin_features))

            return self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior, self.x_swin, self.d_swin, self.reg_loss
        else:
            pass

# x = torch.randn((12, 3, 224, 224)).to(device)
# depth = torch.randn((12, 3, 224, 224)).to(device)
# gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwin(32,3).to(device)
# a, b, c, d, e = model(x,depth, gt)
# print ('done')
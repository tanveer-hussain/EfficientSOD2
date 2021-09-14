import torch
from torch.autograd import Variable
from  ResNet_models_Custom import Saliency_feat_encoder, Encoder_x, Encoder_xy
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpud")
from torch.distributions import Normal, Independent, kl
from main_Residual_swin import SwinIR

class ResSwin(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwin, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        self.xy_encoder = Encoder_xy(7, channel, latent_dim)
        self.x_encoder = Encoder_x(6, channel, latent_dim)
        model_path = "/home/tinu/PycharmProjects/EfficientSOD2/swin_ir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
        self.swinmodel = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        msg = self.swinmodel.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.swinmodel = self.swinmodel.to(device)
        print(msg)

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

            self.x_swin_features = self.swinmodel(x)
            self.d_swin_features = self.swinmodel(depth)

            return self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            pass

x = torch.randn((8, 3, 224, 224)).to(device)
depth = torch.randn((8, 3, 224, 224)).to(device)
gt = torch.randn((8, 1, 224, 224)).to(device)
model = ResSwin(32,3).to(device)
a, b, c, d, e = model(x,depth, gt)
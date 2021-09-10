import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from ResNet import *
from utils import init_weights,init_weights_orthogonal_normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from swin_transformer import SwinTransformer

# swin_model = SwinTransformer().to(device)
# checkpoint = torch.load(r"/home/tinu/PycharmProjects/EfficientSOD2/swin_base_patch4_window7_224_22k.pth", map_location="cpu")
# msg = swin_model.load_state_dict(checkpoint, strict=False)
# print (msg)
from swin_ir import network_swinir
upscale = 4
window_size = 8
height = (224 // upscale // window_size + 1) * window_size
width = (224 // upscale // window_size + 1) * window_size
model_path = "/home/tinu/PycharmProjects/EfficientSOD2/swin_ir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
swin_model = network_swinir.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device)
msg = swin_model.load_state_dict(torch.load(model_path)['params'], strict=True)
print(msg)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Triple_Conv(6, 3)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*7*7, latent_size)
        self.fc2 = nn.Linear(256*7*7, latent_size)
        # self.fc1 = nn.Linear(49*768, latent_size)
        # self.fc2 = nn.Linear(49*768, latent_size)


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, 256*7*7)
        # print(output.size())
        # output = self.tanh(output)
        # output = swin_model(self.conv1(input))
        # output = self.flatten(output)

        mu = self.fc1(output)
        # mu_mean = torch.mean(mu, 0, keepdim=True)
        # mu_std = torch.std(mu, 0, keepdim=True)
        # mu = (mu - mu_mean) / mu_std

        logvar = self.fc2(output)
        # logvar_mean = torch.mean(logvar, 0, keepdim=True)
        # log_std = torch.std(logvar, 0, keepdim=True)
        # logvar = (logvar - logvar_mean) / log_std
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Encoder_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Triple_Conv(7, 3)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*7*7, latent_size)
        self.fc2 = nn.Linear(256*7*7, latent_size)
        # self.fc1 = nn.Linear(49*768, latent_size)
        # self.fc2 = nn.Linear(49*768, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, 256*7*7)
        # output = swin_model(self.conv1(x))
        # output = self.flatten(output)

        mu = self.fc1(output)
        # mu_mean = torch.mean(mu, 0, keepdim=True)
        # mu_std = torch.std(mu, 0, keepdim=True)
        # mu = (mu - mu_mean) / mu_std

        logvar = self.fc2(output)
        # logvar_mean = torch.mean(logvar, 0, keepdim=True)
        # log_std = torch.std(logvar, 0, keepdim=True)
        # logvar = (logvar - logvar_mean) / log_std
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Generator(nn.Module):
    def __init__(self, channel, latent_dim):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.xy_encoder = Encoder_xy(7, channel, latent_dim)
        self.x_encoder = Encoder_x(6, channel, latent_dim)
        self.tanh = nn.Tanh()

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
            return self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            _, mux, logvarx = self.x_encoder(torch.cat((x,depth),1))
            z_noise = self.reparametrize(mux, logvarx)
            self.prob_pred,_  = self.sal_encoder(x,depth,z_noise)
            return self.prob_pred


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)



class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class multi_scale_aspp(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, channel):
        super(multi_scale_aspp, self).__init__()
        self.ASPP_3 = _DenseAsppBlock(input_num=channel, num1=channel * 2, num2=channel, dilation_rate=3,
                                      drop_out=0.1, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=channel * 2, num1=channel * 2, num2=channel,
                                      dilation_rate=6, drop_out=0.1, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=channel * 3, num1=channel * 2, num2=channel,
                                       dilation_rate=12, drop_out=0.1, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=channel * 4, num1=channel * 2, num2=channel,
                                       dilation_rate=18, drop_out=0.1, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=channel * 5, num1=channel * 2, num2=channel,
                                       dilation_rate=24, drop_out=0.1, bn_start=True)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 6, out_channels=channel, kernel_size=1, padding=0)
        )

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        aspp3 = self.ASPP_3(_input)
        feature = torch.cat((aspp3, _input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)

        feature = torch.cat((aspp24, feature), dim=1)

        aspp_feat = self.classification(feature)

        return aspp_feat

######################### copied from SWINIR ######################
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=49, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.unflatten = nn.Unflatten(2,(7,7))

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        # x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1]) # B Ph*Pw C
        x = x.transpose(1,2)
        x = self.unflatten(x)
        return x

    def flops(self):
        flops = 0
        return flops
###########################################

class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.custom_upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(9, 3,3)
        self.convlast = Triple_Conv(3, 1)
        self.conv3 = Triple_Conv(1024, channel)
        self.conv4 = Triple_Conv(2048, channel)

        self.asppconv1 = multi_scale_aspp(channel)
        self.asppconv2 = multi_scale_aspp(channel)
        self.asppconv3 = multi_scale_aspp(channel)
        self.asppconv4 = multi_scale_aspp(channel)

        self.spatial_axes = [2, 3]
        self.conv_depth1 = BasicConv2d(6+latent_dim, 3, kernel_size=3, padding=1)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.conv1_depth = Triple_Conv(768, 768//2)
        self.conv2_depth = Triple_Conv(768//2, 768//8)
        self.conv3_depth = Triple_Conv(768//8, 3)
        self.conv3_depth1 = Triple_Conv(3 , 1)
        # self.conv4_depth = Triple_Conv(2048, channel)

        self.upsampling1 = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.upsampling2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsampling3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        self.layer_depth = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 3, channel * 4)

        if self.training:
            self.initialize_weights()

        img_size = 224
        patch_size = 4
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        embed_dim = 49
        norm_layer = nn.LayerNorm
        self.patch_norm = None

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x,depth,z):
        x_size = (x.shape[2], x.shape[3])
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        x = torch.cat((x, depth, z), 1)
        x = self.conv1(x)

        sal_init = swin_model(x)#.transpose(1,2)
        # sal_init = self.patch_unembed(sal_init, x_size)
        # sal_init = self.conv1_depth(sal_init)
        # sal_init = self.conv2_depth(sal_init)
        # sal_init = self.conv3_depth1(sal_init)
        # sal_init = self.upsampling1(sal_init)
        # sal_init = self.upsampling2(sal_init)
        # sal_init = self.upsampling2(sal_init)

        depth_pred = swin_model(depth)
        # depth_pred = self.patch_unembed(depth_pred, x_size)
        # depth_pred = self.conv1_depth(depth_pred)
        # depth_pred = self.conv2_depth(depth_pred)
        depth_pred = self.conv3_depth(depth_pred)
        # depth_pred = self.upsampling1(depth_pred)
        # depth_pred = self.upsampling2(depth_pred)
        # depth_pred = self.upsampling2(depth_pred)
        # # sal_init = sal_init.transpose(1,2)
        # sal_init = self.upsample4(sal_init)
        # sal_init = self.custom_upsample(sal_init)
        # depth_pred = self.upsample4(depth_pred)
        # depth_pred = self.custom_upsample(depth_pred)
        # depth_pred = self.conv2(depth_pred)
        # print ("x")

        # x = self.conv_depth1(x)
        # x = self.resnet.conv1(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        # x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        # x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        # x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        # x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8
        #
        # ## depth estimation
        # conv1_depth = self.conv1_depth(x1)
        # conv2_depth = self.upsample2(self.conv2_depth(x2))
        # conv3_depth = self.upsample4(self.conv3_depth(x3))
        # conv4_depth = self.upsample8(self.conv4_depth(x4))
        # conv_depth = torch.cat((conv4_depth, conv3_depth, conv2_depth, conv1_depth), 1)
        # depth_pred = self.layer_depth(conv_depth)
        #
        #
        # conv1_feat = self.conv1(x1)
        # conv1_feat = self.asppconv1(conv1_feat)
        # conv2_feat = self.conv2(x2)
        # conv2_feat = self.asppconv2(conv2_feat)
        # conv3_feat = self.conv3(x3)
        # conv3_feat = self.asppconv3(conv3_feat)
        # conv4_feat = self.conv4(x4)
        # conv4_feat = self.asppconv4(conv4_feat)
        # conv4_feat = self.upsample2(conv4_feat)
        #
        # conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        # conv43 = self.racb_43(conv43)
        # conv43 = self.conv43(conv43)
        #
        # conv43 = self.upsample2(conv43)
        # conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        # conv432 = self.racb_432(conv432)
        # conv432 = self.conv432(conv432)
        #
        # conv432 = self.upsample2(conv432)
        # conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        # conv4321 = self.racb_4321(conv4321)
        # conv4321 = self.conv4321(conv4321)
        #
        # sal_init = self.layer6(conv4321)

        return sal_init, depth_pred #self.upsample4(sal_init), self.upsample4(depth_pred)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

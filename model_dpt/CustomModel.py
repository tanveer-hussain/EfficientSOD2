import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import pdb, os, argparse
from datetime import datetime
from ResNet_models import Generator
from data import get_loader
from utils import adjust_lr
from scipy import misc
from utils import l2_regularisation
import smoothness
import imageio
from ResNet import *
import torchvision.models as models
from torch.distributions import Normal, Independent, kl


from dpt.models import DPTSegmentationModel as DPTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_w = net_h = 224
optimize=True
model_path = None
model = DPTModel()
# model = DPTSegmentationModel(
#             150,
#             path=model_path,
#             backbone="vitb_rn50_384",
#         )
model.eval()



if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
model.to(device)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

## load data
image_root = r'D:\My Research\Datasets\Saliency Detection\RGBD\DUT-RGBD\Train/Images/'
gt_root = r'D:\My Research\Datasets\Saliency Detection\RGBD\DUT-RGBD\Train/Labels/'
depth_root = r'D:\My Research\Datasets\Saliency Detection\RGBD\DUT-RGBD\Train/Depth/'
gray_root = r'D:\My Research\Datasets\Saliency Detection\RGBD\DUT-RGBD\Train/Gray/'

train_loader, training_set_size = get_loader(image_root, gt_root, depth_root, gray_root, batchsize=10, trainsize=352)
total_step = len(train_loader)

## define loss
CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
smooth_loss = smoothness.smoothness_loss(size_average=True)


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).sum()


## visualize predictions and gt
def visualize_uncertainty_post_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_post_int.png'.format(kk)
        imageio.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty_prior_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_prior_int.png'.format(kk)
        imageio.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        imageio.imwrite(save_path + name, pred_edge_kk)

## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def kl_divergence(posterior_latent_space, prior_latent_space):
    kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    return kl_div

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)

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

        self.conv1 = Triple_Conv(256, channel)
        self.conv2 = Triple_Conv(512, channel)
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

        self.conv1_depth = Triple_Conv(256, channel)
        self.conv2_depth = Triple_Conv(512, channel)
        self.conv3_depth = Triple_Conv(1024, channel)
        self.conv4_depth = Triple_Conv(2048, channel)
        self.layer_depth = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 3, channel * 4)

        if self.training:
            self.initialize_weights()

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
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        x = torch.cat((x, depth, z), 1)
        x = self.conv_depth1(x).half()
        x = self.resnet.conv1(x).half()
        x = self.resnet.bn1(x).half()
        x = self.resnet.relu(x).half()
        x = self.resnet.maxpool(x).half()
        x1 = self.resnet.layer1(x) .half() # 256 x 64 x 64
        x2 = self.resnet.layer2(x1).half()  # 512 x 32 x 32
        x3 = self.resnet.layer3(x2).half()  # 1024 x 16 x 16
        x4 = self.resnet.layer4(x3).half()  # 2048 x 8 x 8

        ## depth estimation
        conv1_depth = self.conv1_depth(x1).half()
        conv2_depth = self.upsample2(self.conv2_depth(x2)).half()
        conv3_depth = self.upsample4(self.conv3_depth(x3)).half()
        conv4_depth = self.upsample8(self.conv4_depth(x4)).half()
        conv_depth = torch.cat((conv4_depth, conv3_depth, conv2_depth, conv1_depth), 1).half()
        depth_pred = self.layer_depth(conv_depth).half()


        conv1_feat = self.conv1(x1).half()
        conv1_feat = self.asppconv1(conv1_feat).half()
        conv2_feat = self.conv2(x2).half()
        conv2_feat = self.asppconv2(conv2_feat).half()
        conv3_feat = self.conv3(x3).half()
        conv3_feat = self.asppconv3(conv3_feat).half()
        conv4_feat = self.conv4(x4).half()
        conv4_feat = self.asppconv4(conv4_feat).half()
        conv4_feat = self.upsample2(conv4_feat).half()

        conv43 = torch.cat((conv4_feat, conv3_feat), 1).half()
        conv43 = self.racb_43(conv43).half()
        conv43 = self.conv43(conv43).half()

        conv43 = self.upsample2(conv43).half()
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1).half()
        conv432 = self.racb_432(conv432).half()
        conv432 = self.conv432(conv432).half()

        conv432 = self.upsample2(conv432).half()
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1).half()
        conv4321 = self.racb_4321(conv4321).half()
        conv4321 = self.conv4321(conv4321).half()

        sal_init = self.layer6(conv4321).half()

        return self.upsample4(sal_init), self.upsample4(depth_pred)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True).half()
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
        y = self.avg_pool(x).half()
        y = self.conv_du(y).half()
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
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x).half()
        x = self.bn(x).half()
        return x
from  torch.cuda.amp import autocast

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    latent_size = latent_dim = 3
    feat_channel = 32
    sal_encoder = Saliency_feat_encoder(feat_channel, latent_dim).cuda().half()


    print("Let's Play!")
    for epoch in range(1, 40):
        # print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

        for i, pack in enumerate(train_loader, start=1):
            images, gts, depths, grays, index_batch = pack
            # print(index_batch)
            images = Variable(images)
            gts = Variable(gts)
            depths = Variable(depths)
            grays = Variable(grays)
            images = images.cuda().half()
            gts = gts.cuda().half()
            depths = depths.cuda().half()
            grays = grays.cuda().half()

            preprocess_layer_7 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=(3, 3), stride=1, padding=1).cuda().half()
            preprocess_layer_6 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(3, 3), stride=1,
                                           padding=1).cuda().half()


            with torch.no_grad():
                # sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize == True and device == torch.device("cuda"):
                    images = images.to(memory_format=torch.channels_last)

                xy_encoder_input = torch.cat((images,depths,gts),1)
                xy_encoder_input = preprocess_layer_7(xy_encoder_input)

                x_encoder_input = torch.cat((images,depths),1)
                x_encoder_input = preprocess_layer_6(x_encoder_input)



                xy_encoder_output = model.forward(xy_encoder_input)
                x_encoder_output = model.forward(x_encoder_input)

            LinearNet = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                nn.Conv2d(in_channels=64, out_channels=28, kernel_size=(3, 3), stride=1, padding=1),
                nn.AdaptiveAvgPool2d((21, 21))
            ).cuda().half()
            fc1 = nn.Linear(28 * 21 * 21, latent_size).cuda().half()
            fc2 = nn.Linear(28 * 21 * 21, latent_size).cuda().half()

            xy_encoder_output = LinearNet(xy_encoder_output)
            xy_encoder_output = xy_encoder_output.view(xy_encoder_output.size(0), -1)
            muxy = fc1(xy_encoder_output)
            logvarxy = fc2(xy_encoder_output)
            posterior = Independent(Normal(loc=muxy, scale=torch.exp(logvarxy)), 1)

            x_encoder_output = LinearNet(x_encoder_output)
            x_encoder_output = xy_encoder_output.view(x_encoder_output.size(0), -1)
            mux = fc1(x_encoder_output)
            logvarx = fc2(x_encoder_output)
            prior = Independent(Normal(loc=mux, scale=torch.exp(logvarx)), 1)

            lattent_loss = torch.mean(kl_divergence(posterior, prior))

            z_noise_post = reparametrize(muxy, logvarxy)
            z_noise_prior = reparametrize(mux, logvarx)
            with autocast():
                prob_pred_post, depth_pred_post = sal_encoder(images, depths, z_noise_post)
                prob_pred_prior, depth_pred_prior = sal_encoder(images, depths, z_noise_prior)

            smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gts)
            reg_loss = opt.reg_weight * reg_loss
            latent_loss = latent_loss
            depth_loss_post = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_post), depths)
            sal_loss = structure_loss(pred_post, gts) + smoothLoss_post + depth_loss_post
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * latent_loss
            gen_loss_cvae = sal_loss + latent_loss
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae



            print ("Done..!")




             # print(anneal_reg)
            # if epoch % 10 == 0:
            #     opt.lr_gen = opt.lr_gen/10
                # generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999])


        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

        save_path = 'models/'


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % 4 == 0:
            torch.save(generator.state_dict(), save_path + 'DUT_Model' + '_%d' % epoch + '_gen_TRANSFORMER.pth')

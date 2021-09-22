import torch
import torchvision.models as models
import numpy as np
from ResNet import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F

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

from main_Residual_swin import SwinIR
class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        super(Saliency_feat_encoder, self).__init__()

        model_path = "/home/tinu/PycharmProjects/EfficientSOD2/swin_ir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
        self.swinmodel = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        msg = self.swinmodel.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.swinmodel = self.swinmodel.to(device)


        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample14 = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=True)
        self.upsample28 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)
        self.upsample56 = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.conv64 = Triple_Conv(124,64)
        self.conv96 = Triple_Conv(156,96)
        self.conv128 = Triple_Conv(188,128)

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

        self.conv1_depth = Triple_Conv(3, 64)
        self.conv2_depth = Triple_Conv(64, 128)
        self.maxpool1 = nn.MaxPool2d(3,2,0)
        self.conv3_depth = Triple_Conv(256, 512)
        self.conv4_depth = Triple_Conv(2048, channel)
        self.layer_depth = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

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

    def forward(self, x,depth):
        # z = torch.unsqueeze(z, 2)
        # z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        # z = torch.unsqueeze(z, 3)
        # z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        # x = torch.cat((x, depth, z), 1)
        # x = self.conv_depth1(x)
        swin_input = x
        swin_input = torch.nn.functional.interpolate(swin_input, size=64)
        swin_features = self.swinmodel(swin_input)
        # print (swin_features.shape, "swin features shape")
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8

        ## depth estimation
        conv1_depth = self.conv1_depth(depth) # 224 x 224 x 3 = 64 x 64
        conv2_depth = self.upsample2(self.conv2_depth(conv1_depth))
        conv3_depth = self.upsample4(self.conv3_depth(x3))
        conv4_depth = self.upsample8(self.conv4_depth(x4))
        conv_depth = torch.cat((conv4_depth, conv3_depth, conv2_depth, conv1_depth), 1)
        depth_pred = self.layer_depth(conv_depth)


        conv1_feat = self.conv1(x1)
        conv1_feat = self.asppconv1(conv1_feat)
        conv2_feat = self.conv2(x2)
        conv2_feat = self.asppconv2(conv2_feat)
        conv3_feat = self.conv3(x3)
        conv3_feat = self.asppconv3(conv3_feat)
        conv4_feat = self.conv4(x4)
        conv4_feat = self.asppconv4(conv4_feat)

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat, self.upsample14(swin_features)), 1)
        conv43 = self.conv64(conv43)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)


        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat,self.upsample28(swin_features)), 1)
        conv432 = self.conv96(conv432)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)

        conv432 = self.upsample2(conv432)

        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat,self.upsample56(swin_features)), 1)
        conv4321 = self.conv128(conv4321)
        conv4321 = self.racb_4321(conv4321)
        conv4321 = self.conv4321(conv4321)

        sal_init = self.layer6(conv4321)

        return self.upsample4(sal_init), self.upsample4(depth_pred)

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

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder, Triple_Conv, multi_scale_aspp, Classifier_Module, RCAB, BasicConv2d
from Multi_head import MHSA
from dpt.models_custom import DPTSegmentationModel, DPTDepthModel

class BasicTransConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, stride, padding=0, dilation=1):
        super(BasicTransConv2d, self).__init__()
        # nn.Conv2d(in_planes, out_planes,
        #           kernel_size=kernel_size, stride=stride,
        #           padding=padding, dilation=dilation, bias=False)
        self.convT = nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3), stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        x = self.convT(x)
        x = self.bn(x)
        return x

class ResidualAttentionUnit(nn.Module):
    def __init__(self, num_features):
        super(ResidualAttentionUnit, self).__init__()
        self.AttentionBlock = nn.Sequential(
            Triple_Conv(num_features, num_features),
            multi_scale_aspp(num_features),
            multi_scale_aspp(num_features),
            MHSA(num_features, width=56, height=56, heads=4)
        )

    def forward(self, x0, x):
        return x0 + self.AttentionBlock(x)

class RecursiveAttention(nn.Module):
    def __init__(self, in_channels, out_channels, n_recursion, encoder_status):
        super(RecursiveAttention, self).__init__()
        self.initial_conv = Triple_Conv(in_channels,out_channels)
        self.n_recursion = n_recursion
        self.RAU = ResidualAttentionUnit(out_channels)

        self.encoder_status = encoder_status


    def forward(self, x):
        # if self.encoder_status:
        x0 = self.initial_conv(x)
        x = x0
        for i in range(self.n_recursion):
            x = self.RAU(x0, x)
        # else:
        #     x0 = self.convtrans(x)
        #     x = x0
        #     for i in range(self.n_recursion):
        #         x = self.RAU(x0, x)
        return x


class RANet(nn.Module):
    def __init__(self, channel, latent_dim):
        super(RANet, self).__init__()

        model_path = "weights/dpt_hybrid-ade20k-53898607.pt"
        self.dpt_model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
        self.dpt_model.eval()
        self.dpt_model = self.dpt_model.to(memory_format=torch.channels_last)


        features = 256
        non_negative = True
        self.RA = RecursiveAttention(features, channel, 8, True)
        self.transconv1 = BasicTransConv2d(32,16,2,0)
        self.transconv2 = BasicTransConv2d(16, 8,2,0)
        self.transconv3 = BasicTransConv2d(8, 1, 2, 0)
        self.transconv4 = BasicTransConv2d(1, 1, 2, 0)
        self.upsampling224 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        #
        # self.conv43 = Triple_Conv(2 * channel, channel)
        # self.conv432 = Triple_Conv(3 * channel, channel)
        # self.conv4321 = Triple_Conv(4 * channel, channel)
        # self.conv1_1 = Triple_Conv(96, channel)
        # self.conv1_11 = Triple_Conv(64, channel)
        # self.conv1 = Triple_Conv(256, channel)
        # self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        # self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv2 = Triple_Conv(150, 1)
        # self.conv11 = Triple_Conv(6,3)


    def forward(self, x , d, training=True):
        # if training:
        # self.x_sal = self.sal_encoder(x)
        # x = torch.cat((x,d),1)
        # x = self.conv11(x)
        _, _, p2, p3, p4 = self.dpt_model(x) # p1: [2, 256, 112, 112], p2: [2, 256, 56, 56], p3: [2, 256, 28, 28], p4: [2, 256, 14, 14]
        p4 = self.RA(p2)

        p4 = self.transconv1(p4)
        p4 = self.transconv2(p4)
        p4 = self.transconv3(p4)
        p4 = self.transconv4(p4)
        p4 = self.upsampling224(p4)

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
        # conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, self.upsample2(conv1_feat)),
        #                      1)
        # conv4321 = self.racb_4321(conv4321)
        # conv4321 = self.conv4321(conv4321)
        #
        # sal_init = self.layer6(conv4321)
        # out = self.conv2(out)


        return p4 #sal_init# , self.d_sal #self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        # else:
        #     # self.x_sal = self.sal_encoder(x)
        #     # self.x_sal, _ = self.sal_encoder(x, depth)
        #     self.x_sal = self.dpt_model(x)
        #     self.x_sal = self.conv1(self.x_sal)
        #     # _, mux, logvarx = self.x_encoder(torch.cat((x, depth), 1))
        #     # z_noise = self.reparametrize(mux, logvarx)
        #     # self.prob_pred, _ = self.sal_encoder(x, depth, z_noise)
        #     # self.prob_pred, _ = self.sal_encoder(x, depth)
        #     return self.x_sal
    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

# x = torch.randn((2, 3, 224, 224)).to(device)
# depth = torch.randn((2, 3, 224, 224)).to(device)
# # # gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwinModel(32,3).to(device)
# y = model(x,depth)
# print (y.shape)
# print ('done')
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder, Triple_Conv, multi_scale_aspp, Classifier_Module, RCAB, BasicConv2d
from Multi_head import MHSA
from dpt.models_custom import DPTSegmentationModel, DPTDepthModel
import torch.nn.functional as F
from depth_model import DepthNet

class Pyramid_block(nn.Module):
    def __init__(self, in_channels, in_resolution,out_channels,out_resolution,heads,initial):
        super(Pyramid_block, self).__init__()


        self.block1 = nn.ModuleList()

        if in_channels != out_channels:
            self.block1.append(Triple_Conv(in_channels, out_channels))


        if initial==1:
            self.block1.append(MHSA(out_channels, width=in_resolution, height=in_resolution, heads=heads))
            self.block1.append(multi_scale_aspp(in_channels))
        else:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(in_channels, width=in_resolution, height=in_resolution, heads=heads))
        self.block1 = nn.Sequential(*self.block1)


        self.in_resolution = in_resolution
        self.out_resolution = out_resolution

    def forward(self, x):
        x = self.block1(x)
        if self.in_resolution != self.out_resolution:
            x = F.interpolate(x, size=(self.out_resolution,self.out_resolution), mode='bilinear',align_corners=True)

        return x

class ResSwinModel(nn.Module):
    def __init__(self, channel, latent_dim):
        super(ResSwinModel, self).__init__()

        model_path = "weights/dpt_hybrid-midas-501f0c75.pt"
        self.dpt_model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        #
        # model_path = "weights/dpt_hybrid-ade20k-53898607.pt"
        # self.dpt_model = DPTSegmentationModel(
        #     150,
        #     path=model_path,
        #     backbone="vitb_rn50_384",
        # )
        self.dpt_model.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpt_model = self.dpt_model.to(memory_format=torch.channels_last)
        self.depth_model = DepthNet()

        # self.asppconv1 = multi_scale_aspp(channel)
        # self.asppconv2 = multi_scale_aspp(channel)
        # self.asppconv3 = multi_scale_aspp(channel)
        self.asppconv4 = multi_scale_aspp(channel)

        self.spatial_axes = [2, 3]
        self.conv_depth1 = BasicConv2d(6 + latent_dim, 3, kernel_size=3, padding=1)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.aspp_mhsa1_1 = Pyramid_block(32, 56, 32, 56, 4, 1)
        self.aspp_mhsa1_2 = Pyramid_block(32, 56, 32, 56, 4, 2)
        self.aspp_mhsa1_3 = Pyramid_block(32, 56, 32, 56, 4, 3)

        self.aspp_mhsa2_1 = Pyramid_block(32, 56, 32, 28, 4, 1)
        self.aspp_mhsa2_2 = Pyramid_block(32, 28, 32, 28, 4, 2)
        #
        self.aspp_mhsa3_1 = Pyramid_block(32, 28, 32, 14, 4, 1)
        # # self.aspp_mhsa3_2 = Pyramid_block(32, 14, 32, 14, 4, 2)
        #
        self.aspp_mhsa4_1 = Pyramid_block(32, 14, 32, 7, 4, 1)


        # self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        # if optimize == True and device == torch.device("cuda"):
        #     self.dpt_model = self.dpt_model.to(memory_format=torch.channels_last)
        #     self.dpt_model = self.dpt_model.half()
        #     self.sal_encoder = self.sal_encoder.half()

        # self.relu = nn.ReLU(inplace=True)
        # self.swin_saliency = SwinSaliency()
        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)
        self.conv1_1 = Triple_Conv(64, channel)
        self.conv1 = Triple_Conv(256, channel)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = Triple_Conv(150, 1)
        # self.conv3 = Triple_Conv(1024, channel)
        # self.conv4 = Triple_Conv(2048, channel)
        # self.liner1024 = nn.Linear(2048, 1024)
        # self.upsampling = nn.Sequential(
        #     nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        # )


    def forward(self, x , d, training=True):
        # if training:
        # self.x_sal = self.sal_encoder(x)
        _, p1, p2, p3, p4 = self.dpt_model(x)
        d1, d2, d3 = self.depth_model(d)
        # self.x1, self.x2, self.x3, self.x4 = self.sal_encoder(x, self.depth)

        conv1_feat = self.conv1(p1)
        conv1_feat_x1 = F.interpolate(conv1_feat, size=(56,56), mode='bilinear',align_corners=True)
        conv1_feat_x1_d1 = self.conv1_1(torch.cat((conv1_feat_x1,d1),1))
        conv1_feat = self.aspp_mhsa1_1(conv1_feat_x1_d1)
        conv1_feat = self.aspp_mhsa1_2(conv1_feat)
        # conv1_feat = self.aspp_mhsa1_3(conv1_feat)
        conv1_feat = torch.cat((conv1_feat,conv1_feat_x1),1)
        conv1_feat = self.conv1_1(conv1_feat)

        conv2_feat_x2 = self.conv1(p2)
        conv2_feat_x2_d2 = self.conv1_1(torch.cat((conv2_feat_x2,d2),1))
        conv2_feat = self.aspp_mhsa2_1(conv2_feat_x2_d2)
        conv2_feat = self.aspp_mhsa2_2(conv2_feat)
        # conv2_feat = self.asppconv2(conv2_feat)
        conv3_feat = self.conv1(p3)
        d3 = F.interpolate(d3, size=(28,28), mode='bilinear',align_corners=True)
        conv3_feat_x3_d3 = self.conv1_1(torch.cat((conv3_feat,d3),1))
        conv3_feat = self.aspp_mhsa3_1(conv3_feat_x3_d3)
        # conv3_feat = self.asppconv3(conv3_feat)
        conv4_feat = self.conv1(p4)
        conv4_feat = self.aspp_mhsa4_1(conv4_feat)
        # conv4_feat = self.asppconv4(conv4_feat)
        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        conv4321 = self.conv4321(conv4321)

        sal_init = self.layer6(conv4321)
        # out = self.conv2(out)


        return self.upsample4(sal_init) #sal_init# , self.d_sal #self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
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
# print ('done')
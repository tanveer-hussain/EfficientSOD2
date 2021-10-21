import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
from ResNet_models_Custom import Saliency_feat_encoder, Triple_Conv, multi_scale_aspp
from Multi_head import MHSA
from dpt.models_custom import DPTSegmentationModel, DPTDepthModel
import torch.nn.functional as F

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
        model_path = "weights/dpt_hybrid-ade20k-53898607.pt"
        self.dpt_model = DPTSegmentationModel(
            224,
            path=model_path,
            backbone="vitb_rn50_384",
        )
        self.dpt_model.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpt_model = self.dpt_model.to(memory_format=torch.channels_last)
        # self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        # if optimize == True and device == torch.device("cuda"):
        #     self.dpt_model = self.dpt_model.to(memory_format=torch.channels_last)
        #     self.dpt_model = self.dpt_model.half()
        #     self.sal_encoder = self.sal_encoder.half()

        # self.relu = nn.ReLU(inplace=True)
        # self.swin_saliency = SwinSaliency()
        self.conv1_1 = Triple_Conv(64, channel)

        self.conv1 = Triple_Conv(256, channel)
        # self.conv2 = Triple_Conv(512, channel)
        # self.conv3 = Triple_Conv(1024, channel)
        # self.conv4 = Triple_Conv(2048, channel)
        # self.liner1024 = nn.Linear(2048, 1024)
        # self.upsampling = nn.Sequential(
        #     nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
        # nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        # )


    def forward(self, x , training=True):
        # if training:
        # self.x_sal = self.sal_encoder(x)
        self.p1, self.p2, self.p3, self.p4 = self.dpt_model(x)
        # self.x1, self.x2, self.x3, self.x4 = self.sal_encoder(x, self.depth)

        conv1_feat = self.conv1(self.p1)
        conv1_feat = self.aspp_mhsa1_1(conv1_feat)
        conv1_feat = self.aspp_mhsa1_2(conv1_feat)
        # conv1_feat = self.aspp_mhsa1_4(conv1_feat)
        conv1_feat = self.conv1_1(torch.cat((self.conv1(self.p1), conv1_feat), 1))
        # print (conv1_feat.shape)

        conv2_feat = self.conv1(self.p2)
        conv2_feat = self.aspp_mhsa2_1(conv2_feat)
        conv2_feat = self.aspp_mhsa2_2(conv2_feat)
        conv2_feat = self.conv1_1(torch.cat((self.conv2(self.p2), conv2_feat), 1))

        conv3_feat = self.conv1(self.p3)
        conv3_feat = self.aspp_mhsa3_1(conv3_feat)
        conv3_feat = self.aspp_mhsa3_2(conv3_feat)
        conv3_feat = self.conv1_1(torch.cat((self.conv3(self.p3), conv3_feat), 1))

        conv4_feat = self.conv1(self.p4)
        conv4_feat = self.aspp_mhsa4_1(conv4_feat)
        conv4_feat = self.conv1_1(torch.cat((self.conv4(self.p4), conv4_feat), 1))

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

# x = torch.randn((12, 3, 224, 224)).to(device)
# # depth = torch.randn((12, 3, 224, 224)).to(device)
# # gt = torch.randn((12, 1, 224, 224)).to(device)
# model = ResSwinModel(32,3).to(device)
# y = model(x)
# print ('done')
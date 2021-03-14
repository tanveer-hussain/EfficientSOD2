from torch import nn
import torchvision
from deformable_conv import DeformConv2d
import torch
import cv2
import numpy as np
import torch.nn.functional as F

class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()

        # ******************** Encoding image ********************

        originalmodel = torchvision.models.densenet169(pretrained=False, progress=True)
        # pretrained_model = models.vgg16(pretrained=True).features
        self.custom_model = nn.Sequential(*list(originalmodel.features.children())[:-5])

        self.preprocessLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.AdaptiveAvgPool2d((28,28))
            # nn.Upsample(size=(28 , 28), mode='bilinear', align_corners=True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            )

        self.deform1 = DeformConv2d(64, 128, 3, padding=1, modulation=True)
        self.deform2 = DeformConv2d(128, 128, 3, padding=1, modulation=True)
        # nn.ReLU(),
        self.deform3 = DeformConv2d(128, 64, 3, padding=1, modulation=True)

        # ******************** Decoding image ********************
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        self.upsampling1 = nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True)

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(3, 3), stride=2, padding=1)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(3, 3))
        self.upsampling2 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)



    def forward(self, x, d):
        x = self.custom_model(x)
        x = self.layer1(x)
        d = self.preprocessLayer(d)
        # print(f"x shape after custom layer > {x.shape, d.shape}")
        x = torch.cat((x, d), 1)
        # print(f"x shape after custom layer > {x.shape, d.shape}")



        x = self.deform1(x)
        x = self.deform2(x)
        x = self.deform3(x)
        # print(f"x shape after custom layer > {x.shape, d.shape}")




        # # ******************** Decoding image ********************
        x = self.deconv1(x)
        x = self.deconv2(x)

        x = self.upsampling1(x)
        x = self.deconv3(x)
        x = self.upsampling2(x)

        # x = torch.cat((x,d), 1)

        # x = torch.cat((x,d), 1)
        # x = self.deconv4(x)
        # x = self.upsampling2(x)
             
        # edges = cv2.Canny(d,100,200)
        # edges = Image.fromarray(edges)

        # x = cv2.addWeighted(d, 0.3, x, 0.7,0)


        return x

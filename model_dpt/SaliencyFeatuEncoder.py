
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
        x = self.conv_depth1(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8

        ## depth estimation
        conv1_depth = self.conv1_depth(x1)
        conv2_depth = self.upsample2(self.conv2_depth(x2))
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
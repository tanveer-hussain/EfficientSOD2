import torch
import torch.nn.functional as F
from torch.autograd import Variable

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


from dpt.models import DPTSegmentationModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_w = net_h = 224
optimize=True
model_path = None
model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
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

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    latent_size = 3

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
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            grays = grays.cuda()

            with torch.no_grad():
                # sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize == True and device == torch.device("cuda"):
                    images = images.to(memory_format=torch.channels_last)
                    images = images.half()

                out = model.forward(images)

            customNet = nn.Sequential(
                nn.Conv2d(in_channels=150, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                nn.Conv2d(in_channels=64, out_channels=28, kernel_size=(3, 3), stride=1, padding=1),
                nn.AdaptiveAvgPool2d((21, 21)),
                nn.Linear(28 * 21 * 21, latent_size)
            )



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

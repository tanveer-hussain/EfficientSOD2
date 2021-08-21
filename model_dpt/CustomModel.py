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
from torch.distributions import Normal, Independent, kl


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

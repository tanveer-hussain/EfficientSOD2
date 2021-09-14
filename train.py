import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
# from ResNet_models import Generator
from ResNet_models_UCNet import Generator
from data import get_loader
from utils import adjust_lr
from utils import l2_regularisation
import smoothness
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--sm_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
parser.add_argument('--lat_weight', type=float, default=10.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--depth_loss_weight', type=float, default=0.1, help='weight for depth loss')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
# generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
# generator.cuda()
#
# generator_params = generator.parameters()
# generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999])

from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=opt.feat_channel, latent_dim=opt.latent_dim)
resswin.cuda()
resswin_params = resswin.parameters()
resswin_optimizer = torch.optim.Adam(resswin_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999])
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
        # print ('pred_edge_kk shape', pred_edge_kk.shape)
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
        # print('proir_edge_kk shape', pred_edge_kk.shape)
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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# dataset_name = datasets[5]
from  ResNet_models_Custom import Triple_Conv
import torch.nn as nn
from main_Residual_swin import SwinIR
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
if __name__ == '__main__':

    model_path = "/home/tinu/PycharmProjects/EfficientSOD2/swin_ir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    swinmodel = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                            mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    msg = swinmodel.load_state_dict(torch.load(model_path)['params'], strict=True)
    swinmodel = swinmodel.to(device)

    TrippleConv1 = Triple_Conv(60, 30).cuda()
    TrippleConv2 = Triple_Conv(30, 1).cuda()
    upsample3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False).cuda()
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True).cuda()
    print(msg)


    torch.multiprocessing.freeze_support()
    print("Let's Play!")
    ## load data
    datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
    save_results_path = r"/home/tinu/PycharmProjects/EfficientSOD2/TempResults.dat"
    save_path = 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for dataset_name in datasets:
        print ("Datasets:", datasets, "\n ****Currently Training > ", dataset_name)
        image_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Images/'
        gt_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Labels/'
        depth_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Depth/'
        gray_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Gray/'
        train_loader, training_set_size = get_loader(image_root, gt_root, depth_root, gray_root,batchsize=opt.batchsize, trainsize=opt.trainsize)
        total_step = len(train_loader)

        for epoch in range(1, opt.epoch+1):
            for i, pack in enumerate(train_loader, start=1):
                images, gts, depths, grays, index_batch = pack
                # print(index_batch)
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                depths = Variable(depths).cuda()
                grays = Variable(grays).cuda()
                pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior, reg_loss = resswin.forward(images,depths,gts)

                x = F.interpolate(images, size=64)
                depth = F.interpolate(depths, size=64)
                x_swin_features = swinmodel(x)
                d_swin_features = swinmodel(depth)

                x_swin_features = TrippleConv2(TrippleConv1(x_swin_features))
                d_swin_features = TrippleConv2(TrippleConv1(d_swin_features))

                x_swin = upsample(upsample3(x_swin_features))
                d_swin = upsample(upsample3(d_swin_features))


                smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gts)
                reg_loss = opt.reg_weight * reg_loss
                latent_loss = latent_loss
                depth_loss_post = opt.depth_loss_weight*mse_loss(torch.sigmoid(depth_pred_post),depths)
                sal_loss = structure_loss(pred_post, gts) + smoothLoss_post + depth_loss_post
                anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
                latent_loss = opt.lat_weight*anneal_reg *latent_loss
                gen_loss_cvae = sal_loss + latent_loss
                gen_loss_cvae = opt.vae_loss_weight*gen_loss_cvae

                smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gts)
                depth_loss_prior = opt.depth_loss_weight*mse_loss(torch.sigmoid(depth_pred_prior),depths)
                gen_loss_gsnn = structure_loss(pred_prior, gts) + smoothLoss_prior + depth_loss_prior
                gen_loss_gsnn = (1-opt.vae_loss_weight)*gen_loss_gsnn
                gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss

                resswin_optimizer.zero_grad()
                gen_loss.backward()
                resswin_optimizer.step()
                visualize_gt(gts)
                visualize_uncertainty_post_init(torch.sigmoid(x_swin))
                visualize_uncertainty_prior_init(torch.sigmoid(pred_prior))

                if i % 50 == 0 or i == total_step:
                    print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen vae Loss: {:.4f}, gen gsnn Loss: {:.4f}, reg Loss: {:.4f}'.
                        format(epoch, opt.epoch, i, total_step, gen_loss_cvae.data, gen_loss_gsnn.data, reg_loss.data))

            adjust_lr(resswin_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
            if epoch % 50 == 0:
                with open(save_results_path, "a+") as ResultsFile:
                    writing_string = dataset_name + "  Epoch [" + str(epoch) + "/" + str(opt.epoch) + "] Step [" + str(i) + "/" + str(total_step) + "], gen vae Loss:" + str(round(gen_loss_cvae.data.item(),4)) + ", gen_loss_gsnn:" + str(round(gen_loss_gsnn.data.item(),4)) + ", reg_loss:" + str(round(reg_loss.data.item(),4)) + "\n"
                    ResultsFile.write(writing_string)
                torch.save(resswin.state_dict(), save_path + dataset_name + 'SWIN' + '_%d' % epoch + '_UCNet.pth')

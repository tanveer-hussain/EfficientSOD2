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
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
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
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999])

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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# dataset_name = datasets[5]


if __name__ == '__main__':
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
                pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator.forward(images,depths,gts)

                reg_loss = l2_regularisation(generator.xy_encoder) + \
                        l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
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

                generator_optimizer.zero_grad()
                gen_loss.backward()
                generator_optimizer.step()
                visualize_gt(gts)
                visualize_uncertainty_post_init(torch.sigmoid(pred_post))
                visualize_uncertainty_prior_init(torch.sigmoid(pred_prior))

                if i % 50 == 0 or i == total_step:
                    print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen vae Loss: {:.4f}, gen gsnn Loss: {:.4f}, reg Loss: {:.4f}'.
                        format(epoch, opt.epoch, i, total_step, gen_loss_cvae.data, gen_loss_gsnn.data, reg_loss.data))

            adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
            if epoch % 50 == 0:
                with open(save_results_path, "a+") as ResultsFile:
                    writing_string = dataset_name + "  Epoch [" + str(epoch) + "/" + str(opt.epoch) + "] Step [" + str(i) + "/" + str(total_step) + "], gen vae Loss:" + str(round(gen_loss_cvae.data.item(),4)) + ", gen_loss_gsnn:" + str(round(gen_loss_gsnn.data.item(),4)) + ", reg_loss:" + str(round(reg_loss.data.item(),4)) + "\n"
                    ResultsFile.write(writing_string)
                torch.save(generator.state_dict(), save_path + dataset_name + 'SWIN' + '_%d' % epoch + '_UCNet.pth')

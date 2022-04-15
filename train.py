import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
from data import TrainDatasetLoader
from utils import adjust_lr
from utils import l2_regularisation
import smoothness
import imageio
import torch.nn as nn
from customlosses import ssim
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## define loss

criterion = nn.MSELoss().to('cuda')
CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
smooth_loss = smoothness.smoothness_loss(size_average=True)
l1_criterion = nn.L1Loss()

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
        name = '{:02d}_depth.png'.format(kk)
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


from PASModel import PASNet
device = torch.device('cuda' if torch.cuda.is_available else "cpu")

if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()
    ######################### Inputs ############################
    rgbd_datasets = ['SIP', "DUT-RGBD", "NLPR", 'NJU2K']
    rgb_datasets = ["DUTS-TE", "ECSSD", 'HKU-IS', 'Pascal-S']
    save_results_path = r"/home/tinu/PycharmProjects/EfficientSOD2/TempResults.dat"
    save_path = 'models/'
    ## Hyper parameters
    epochs = 2
    batchsize = 6
    lr = 5e-5
    decay_rate = 0.9
    decay_epoch = 20
    beta1_gen = 0.5
    # weight_decay = 0.001
    feature_channels = 32
    dim = 3
    # smoothness_weight = 0.1
    regularization_weight = 1e-4

    ############################################################

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for dataset_name in rgbd_datasets:

        PASNet = PASNet(channel=feature_channels, latent_dim=dim)
        PASNet.to(device)
        PASNet.train()
        resswin_params = PASNet.parameters()
        resswin_optimizer = torch.optim.Adam(resswin_params, lr, betas=[beta1_gen, 0.999])
        print ("Datasets:", rgbd_datasets, "\n ****Currently Training > ", dataset_name)
        dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/' + dataset_name
        d_type = ['Train', 'Test']

        train_data = TrainDatasetLoader(dataset_path, d_type[0])
        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
        total_step = len(train_loader)

        for epoch in range(1, epochs):
            for i, (images, depths, _, _, gts, _) in enumerate(train_loader, start=1):

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                depths = Variable(depths).cuda()

                # x_sal, d_sal = resswin.forward(images, depths)
                x_sal = PASNet.forward(images, depths)
                # x_sal = torch.sigmoid(x_sal)
                # total_loss = mse_loss(x_sal,gts)
                # reg_loss = l2_regularisation(resswin.sal_encoder)
                reg_loss = l2_regularisation(PASNet.dpt_model) #+ l2_regularisation(resswin.dpt_depth_model)
                reg_loss = regularization_weight * reg_loss
                #
                #depth_loss = l1_criterion(d_sal, gts)
                #d_ssim_loss = torch.clamp((1 - ssim(d_sal, gts, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
                #
                # sal_loss = CE(x_sal, gts)
                x_ssim_loss = torch.sigmoid(torch.clamp((1 - ssim(x_sal, gts, val_range=1000.0 / 10.0)) * 0.5, 0, 1))
                #
                #x_loss = (0.2 * structure_loss(x_sal, gts)) + (0.3 * smooth_loss(torch.sigmoid(x_sal), gts)) + (0.3 * x_ssim_loss) + (0.2 * sal_loss)
                #d_loss = (0.2 * structure_loss(d_sal, gts)) + (0.3 * smooth_loss(torch.sigmoid(d_sal), gts))  + (0.3 * d_ssim_loss) + (0.2 * depth_loss)
                #
                # anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
                total_loss = criterion(x_sal,gts) + reg_loss#x_ssim_loss + reg_loss # + x_loss # + d_loss

                #
                resswin_optimizer.zero_grad()
                total_loss.backward()
                resswin_optimizer.step()
                visualize_uncertainty_post_init(torch.sigmoid(x_sal))
                visualize_uncertainty_prior_init(torch.sigmoid(gts))
                #
                if i % 2 == 0 or i == total_step:
                    print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen vae Loss: {:.4f}'.
                        format(epoch, epochs, i, total_step, total_loss.data))
                    print("Dataset: ", dataset_name)

            adjust_lr(resswin_optimizer, lr, epoch, decay_rate, decay_epoch)
            if epoch % 20 == 0 or epoch == epochs:
                torch.save(PASNet.state_dict(), save_path + dataset_name + 'RGBD_D' + '_%d' % epoch + '_Pyramid.pth')
                # with open(save_results_path, "a+") as ResultsFile:
                #     writing_string = dataset_name + "  Epoch [" + str(epoch) + "/" + str(opt.epoch) + "] Step [" + str(i) + "/" + str(total_step) + "], Loss:" + str(round(total_loss.data.item(),4))  + "\n"
                #     ResultsFile.write(writing_string)
        image_save_path = 'results/' + dataset_name + "/"
        image_save_path if os.path.exists(image_save_path) else os.mkdir(image_save_path)
        test_data = TrainDatasetLoader(dataset_path, d_type[1])
        test_loader = DataLoader(test_data)
        from test import ModelTesting

        ModelTesting(PASNet, test_loader, image_save_path, dataset_path, dataset_name)
        torch.cuda.empty_cache()

from multiprocessing.dummy import freeze_support
import torch
from data import test_dataset
import numpy as np
import cv2
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available else "cpu")
latent_dim=3
feat_channel=32
test_datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
# test_datasets = ['NJU2K', 'SIP']
# dataset_name = datasets[0]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/'# + dataset_name
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
epoch = 100
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
resswin.to(device)
import os

for dataset in test_datasets:
    save_path = 'results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    resswin.load_state_dict(torch.load("models/" + dataset + 'RGB' + '_%d' % epoch + '_Pyramid.pth'))
    # resswin.load_state_dict(torch.load("models/" + 'NJU2KRGB_50_Pyramid.pth'))

    print('Model loaded')
    resswin.eval()

    image_root = dataset_path + dataset + "/Test" + '/Images/'
    depth_root = dataset_path + dataset + "/Test" + '/Depth_Synthetic/'
    test_loader = test_dataset(image_root, depth_root, 352)
    for i in range(test_loader.size):
        # print (i)
        image, HH, WW, name = test_loader.load_data()
        # image, depth, HH, WW, name = test_loader.load_data()

        print ("Processing..", image_root + name)
        image = image.cuda()
        # depth = depth.cuda()
        output = resswin.forward(image, training=False)
        # res = output
        # res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # imsave(save_path+name, res)
        # print (res.shape)

        output = torch.squeeze(output, 0)
        output = output.detach().cpu().numpy()
        output = output.dot(255)
        output *= output.max() / 255.0
        output = np.transpose(output, (1, 2, 0))
        output_path = save_path + name
        cv2.imwrite(output_path, output)
        # misc.imsave(save_path + name, output)
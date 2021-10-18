from multiprocessing.dummy import freeze_support
import torch
from data import test_dataset
import numpy as np
import cv2
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available else "cpu")
latent_dim=3
feat_channel=32
# test_datasets = ['DUT-RGBD', "NJU2K"]#, "NLPR", 'SIP']
test_datasets = ['DUT-RGBD']
# dataset_name = datasets[0]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/'# + dataset_name
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
epoch = 30
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
resswin.to(device)
import os


for dataset in test_datasets:
    save_path = 'results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    resswin.load_state_dict(torch.load("models/" + dataset + 'RGBD' + '_%d' % epoch + '_Pyramid.pth'))
    # resswin.load_state_dict(torch.load("models/" + 'NJU2KRGB_50_Pyramid.pth'))

    print('Model loaded')
    resswin.eval()

    image_root = dataset_path + dataset + "/Test" + '/Images/'
    depth_root = dataset_path + dataset + "/Test" + '/Depth/'
    test_loader = test_dataset(image_root, depth_root, 352)
    for i in range(test_loader.size):
        # print (i)
        image, HH, WW, name = test_loader.load_data()

        print ("Processing..", image_root + name)
        image = image.cuda()
        # depth = depth.cuda()
        # output = resswin.forward(image, depth, training=False)
        output = resswin.forward(image, training=False)

        output = F.interpolate(output, size=(HH, WW), mode='bilinear', align_corners=False)
        output = torch.squeeze(output, 0)
        output = output.detach().cpu().numpy()
        output = output.dot(255)
        output *= output.max() / 255.0
        output = np.transpose(output, (1, 2, 0))
        output_path = save_path + name
        cv2.imwrite(output_path, output)
        # misc.imsave(save_path + name, output)

class ModelTesting():


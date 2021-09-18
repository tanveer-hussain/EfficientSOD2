import torch
import torch.nn.functional as F
import pdb, os, argparse
# from ResNet_models_UCNet import Generator
from ResNet_models import Generator
from data import test_dataset

import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
dataset_name = datasets[3]
dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'

epoch = 100
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=opt.feat_channel, latent_dim=opt.latent_dim)
resswin.cuda()
resswin.load_state_dict(torch.load("models/" + dataset_name+ 'SWIN' + '_%d' % epoch + '_UCNet.pth'))
print ('Model loaded')
resswin.eval()


save_path = r'/home/tinu/PycharmProjects/EfficientSOD2/output'
image_root = dataset_path + '/Images/'
depth_root = dataset_path + '/Depth/'
print (image_root, "\n", depth_root)
test_loader = test_dataset(image_root, depth_root, opt.testsize)
for i in range(test_loader.size):

    image, depth, HH, WW, name = test_loader.load_data()


    image = image.cuda()
    depth = depth.cuda()

    import timeit

    start_time = timeit.default_timer()
    generator_pred = resswin.forward(image, depth, training=False)
    #print('Single prediction time consumed >> , ', timeit.default_timer() - start_time, ' seconds')
    res = generator_pred
    res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    # name = name[:-3]
    output_path = os.path.join(save_path,name)
    # cv2.imshow('',res)
    # cv2.waitKey(0)
    print(output_path)

    cv2.imwrite(output_path, res*255)
    # res.save(save_path+name, res)

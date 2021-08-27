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

datasets = ["DUT-RGBD", "NLPR", 'DES', 'LFSD','NJU2K','NLPR','SIP','STERE']
dataset_name = datasets[4]
dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'

epoch = 30
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load("models/" + dataset_name+ '_SWINModel' + '_%d' % epoch + '_UCNet.pth'))
print ('Model loaded')
generator.cuda()
generator.eval()


# test_datasets = ['DES', 'LFSD','NJU2K','NLPR','SIP','STERE']
#test_datasets = ['STERE']

# for dataset in test_datasets:
#     save_path = crossdata_output
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

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
    generator_pred = generator.forward(image, depth, training=False)
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

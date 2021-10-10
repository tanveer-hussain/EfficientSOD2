from multiprocessing.dummy import freeze_support

import torch
import torch.nn.functional as F
import os, argparse
from torch.utils.data import Dataset, DataLoader
from data import DatasetLoader
import cv2
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
# print (device)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
dataset_name = datasets[0]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/' + dataset_name
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'

epoch = 100
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=opt.feat_channel, latent_dim=opt.latent_dim)
resswin.to(device)
resswin.load_state_dict(torch.load("models/" + dataset_name+ 'SD' + '_%d' % epoch + '_.pth'))
print ('Model loaded')
resswin.eval()


# save_path = r'/home/tinu/PycharmProjects/EfficientSOD2/output'
save_path = r"output"
image_root = dataset_path + '/Images/'
depth_root = dataset_path + '/Depth_Synthetic/'
print (image_root, "\n", depth_root)
d_type = ['Train', 'Test']
test_data = DatasetLoader(dataset_path, d_type[1])
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=16, drop_last=True)

if __name__ == '__main__':
    freeze_support()
    for i, (images, depths, gts) in enumerate(test_loader, start=1):
        #
        # image, depth, HH, WW, name = test_loader.load_data()


        images = images.to(device)
        depths = depths.to(device)

        import timeit

        start_time = timeit.default_timer()
        generator_pred = resswin.forward(images, depths, training=False)
        #print('Single prediction time consumed >> , ', timeit.default_timer() - start_time, ' seconds')
        print (generator_pred.shape)
        res = generator_pred
        res = F.upsample(res, size=[224,224], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # name = name[:-3]
        output_path = os.path.join(save_path,name)
        # cv2.imshow('',res)
        # cv2.waitKey(0)
        print(output_path)

        cv2.imwrite(output_path, res*255)
        # res.save(save_path+name, res)
main()
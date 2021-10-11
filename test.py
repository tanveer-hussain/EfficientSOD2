from multiprocessing.dummy import freeze_support
import torch
from data import test_dataset
from scipy import misc
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
latent_dim=3
feat_channel=32
test_datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
# dataset_name = datasets[0]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/'# + dataset_name
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
epoch = 50
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
resswin.to(device)
import os

for dataset in test_datasets:
    save_path = 'results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    resswin.load_state_dict(torch.load("models/" + dataset + 'SD' + '_%d' % epoch + '_.pth'))
    print('Model loaded')
    resswin.eval()

    image_root = dataset_path + dataset + "/Test" + '/Images/'
    depth_root = dataset_path + dataset + "/Test" + '/Depth_Synthetic/'
    test_loader = test_dataset(image_root, depth_root, 352)
    for i in range(test_loader.size):
        print (i)
        image, depth, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        generator_pred = resswin.forward(image, depth, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        misc.imsave(save_path+name, res)
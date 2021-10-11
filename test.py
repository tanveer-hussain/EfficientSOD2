from multiprocessing.dummy import freeze_support
import torch
from torchvision import transforms as T

from data import TestDatasetLoader
device = torch.device('cuda' if torch.cuda.is_available else "cpu")

latent_dim=3
feat_channel=32

datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
dataset_name = datasets[0]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/' + dataset_name
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
# DUT-RGBDSD_50_
epoch = 50
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
resswin.to(device)
resswin.load_state_dict(torch.load("models/" + dataset_name + 'SD' + '_%d' % epoch + '_.pth'))
print ('Model loaded')
resswin.eval()


def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)
    return x

# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name
d_type = ['Train', 'Test']

# image_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Images/'
# gt_root = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Train/Labels/'

test_data = TestDatasetLoader(dataset_path, d_type[0])
train_loader = DataLoader(test_data, batch_size=opt.batchsize, shuffle=True, num_workers=16, drop_last=True)

import os
dataset_name = datasets[0]
input_dir = r'D:\My Research\Datasets\Saliency Detection\RGBD' + dataset_name + "\Images"
output_dir = r'C:\Users\user02\Documents\GitHub\EfficientSOD2\Output'  + dataset_name + "\\"
output_dir = output_dir if os.path.exists(output_dir) else os.mkdir(output_dir)

testing_code_dir(input_dir,output_dir)
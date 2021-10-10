from multiprocessing.dummy import freeze_support
import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import cv2
import timeit
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
# print (device)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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



datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']


def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)
    return x
def predictions(img):

    x = preprocess_image(img)
    start_time = timeit.default_timer()
    output = resswin(x)
    output = torch.squeeze(output, 0)

    output = output.detach().cpu().numpy()
    output = output.dot(255)
    output *= output.max()/255.0
    # print (max(output))
    # output = cv2.erode(output, kernel, iterations=2)
    # output = cv2.dilate(output, kernel, iterations=1)
    return output

def testing_code_dir(input_dir, output_dir):

    val_base_path_images = os.listdir(input_dir)
    for single_image in val_base_path_images:
        full_path = input_dir + single_image

        img = Image.open(full_path).convert("RGB")

        output = predictions(img)
        output = np.transpose(output, (1, 2, 0))
        # cv2.imshow('', output)
        # cv2.waitKey(50)

        output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
        cv2.imwrite(output_path, output)
        print("Reading: %s\n writing: %s " % (full_path, output_path))


import os
dataset_name = datasets[0]
input_dir = r'D:\My Research\Datasets\Saliency Detection\RGBD' + dataset_name + "\Images"
output_dir = r'C:\Users\user02\Documents\GitHub\EfficientSOD2\Output'  + dataset_name + "\\"
output_dir = output_dir if os.path.exists(output_dir) else os.mkdir(output_dir)

testing_code_dir(input_dir,output_dir)
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
import torch

class TrainDatasetLoader(Dataset):

    def __init__(self, dir, d_type):
        self.x_path = os.path.join(dir, str(d_type), 'Images')
        self.y_path = os.path.join(dir, str(d_type), 'Labels')
        # self.d_path = os.path.join(dir, str(d_type), 'Depth')

        # print (self.x_path, "\n", self.y_path, "\n", self.d_path)

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)
        # self.D = os.listdir(self.d_path)

        self.length = len(self.X)

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])

        ###################### depth estimation
        # self.return_depth = return_depth
        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        # self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.midas.eval()
        # self.transform = self.midas_transforms.default_transform


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])
        # d_full_path = os.path.join(self.d_path, self.D[index])

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')
        # d = Image.open(d_full_path).convert('RGB')


        x = self.img_transform(x)
        y = self.gt_transform(y)
        # d = self.gt_transform(d)

        # print ('x', x.shape, ', y', y.shape, ', d', d.shape)


        return x , y

from torch.utils import data
from ResSwin import ResSwinModel
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
class TestDatasetLoader(data.Dataset):

    def __init__(self, dir, d_type, weights_path):
        self.x_path = os.path.join(dir, str(d_type), 'Images')
        self.y_path = os.path.join(dir, str(d_type), 'Labels')

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)

        self.resswin = ResSwinModel(channel=32, latent_dim=3)
        self.resswin.to(device)
        print(self.resswin.load_state_dict(torch.load(weights_path)))
        # self.D = os.listdir(self.d_path)

        self.length = len(self.X)

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])


        # d_full_path = os.path.join(self.d_path, self.D[index])

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')
        x = self.img_transform(x)

        pred = self.resswin.forward(x, training=False)

        pred = F.interpolate(pred, size=(x.shape[1], x.shape[2]), mode='bilinear', align_corners=False)
        pred = torch.squeeze(pred, 0)
        pred = pred.detach().cpu().numpy()
        pred = pred.dot(255)
        pred *= pred.max() / 255.0
        pred = np.transpose(pred, (1, 2, 0))


        y = np.array(y)

        h = x.size[0]
        w = x.size[1]
        # d = Image.open(d_full_path).convert('RGB')





        return pred , y


class test_dataset:
    def __init__(self, image_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.depth_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name #image, depth, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainDatasetLoader(Dataset):

    def __init__(self, dir, d_type):
        self.x_path = os.path.join(dir, str(d_type), 'Images')
        self.y_path = os.path.join(dir, str(d_type), 'Labels')
        self.d_path = os.path.join(dir, str(d_type), 'Depth_Synthetic')

        # print (self.x_path, "\n", self.y_path, "\n", self.d_path)

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)
        self.D = os.listdir(self.d_path)

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
        d_full_path = os.path.join(self.d_path, self.D[index])

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')
        d = Image.open(d_full_path).convert('RGB')


        x = self.img_transform(x)
        y = self.gt_transform(y)
        d = self.gt_transform(d)

        # print ('x', x.shape, ', y', y.shape, ', d', d.shape)


        return x , d, y


class TestDatasetLoader(Dataset):

    def __init__(self, dir, d_type):
        self.x_path = os.path.join(dir, str(d_type), 'Images')
        self.d_path = os.path.join(dir, str(d_type), 'Depth_Synthetic')

        # print (self.x_path, "\n", self.y_path, "\n", self.d_path)

        self.X = os.listdir(self.x_path)
        self.D = os.listdir(self.d_path)

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
        d_full_path = os.path.join(self.d_path, self.D[index])

        x = Image.open(x_full_path).convert('RGB')
        d = Image.open(d_full_path).convert('RGB')


        x = self.img_transform(x)
        d = self.gt_transform(d)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        # print ('x', x.shape, ', y', y.shape, ', d', d.shape)


        return x , d , name


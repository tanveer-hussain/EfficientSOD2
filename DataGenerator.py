from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn
import numpy as np
import csv
import cv2

class DatasetLoader(Dataset):


    def __init__(self, dir, d_type):

        self.x_path = os.path.join(dir, str(d_type), 'Images')

        self.y_path = os.path.join(dir, str(d_type), 'Labels')

        self.d_path = os.path.join(dir, str(d_type), 'Depth')

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)
        self.D = os.listdir(self.d_path)

        self.length = len(self.X)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])
        d_full_path = os.path.join(self.d_path, self.D[index])

        # print (f'x > {x_full_path}, y > {y_full_path}, d > {d_full_path}')

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')
        d = cv2.imread(d_full_path, 0)
        # cv2.imshow('', d)
        d = Image.fromarray(d)
        # d.show()
        # d = Image.open(d_full_path).convert('RGB')

        # d.show()

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        x = transform(x)
        y = transform(y)
        d = transform(d)


        return x , y , d





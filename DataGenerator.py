from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn

class DatasetLoader(Dataset):


    def __init__(self, dataset_path):
        
        main_file = open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Train.lst", "r")

        main_data = main_file.read().split("\n")
        main_file.close()
        X = []
        Y = []

        self.x_path = os.path.join(dataset_path, 'Images')
        self.y_path = os.path.join(dataset_path, 'Labels')

        for single_image_path in os.listdir(self.x_path):
            X.append(single_image_path)

        for single_image_path in os.listdir(self.y_path):
            Y.append(single_image_path)

        self.X = X
        self.Y = Y
        self.length = len(self.X)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])

        # print (x_full_path)
        # print (y_full_path)

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        x = transform(x)
        y = transform(y)


        return x , y






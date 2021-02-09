from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn
import csv

class DatasetLoader(Dataset):


    def __init__(self, dataset_path, X, Y):

        with open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Train.csv", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x_image = dataset_path + "\Images\\" + str(row[0])
                y_label = dataset_path + "\Labels\\" + str(row[0])[0:(len(str(row[0])) - 3)] + "png"

                X.append(x_image)
                Y.append(y_label)


        # main_file.close()
        # # print (main_data)

        # for single_image_path in main_data:
        #     print (single_image_path)
        #     if not single_image_path is None:
        #         full_path = dataset_path + "\Images\\" + single_image_path[0:(len(single_image_path) - 3)]
        #         X.append(full_path)

        # for single_image_path in main_data:
        #     if not single_image_path is None:
        #         full_path = dataset_path + "\Labels\\" + single_image_path[0:(len(single_image_path) - 3)]
        #         Y.append(full_path)

        self.X = X
        self.Y = Y
        self.length = len(self.X)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = self.X[index]
        y_full_path = self.Y[index]


        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        x = transform(x)
        y = transform(y)


        return x , y






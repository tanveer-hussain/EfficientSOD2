import os
from PIL import Image
import cv2
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class DatasetLoader(Dataset):

    def __init__(self, dir, d_type):

        self.x_path = os.path.join(dir, str(d_type), 'Images')
        self.y_path = os.path.join(dir, str(d_type), 'Labels')

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)

        self.length = len(self.X)

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])

        ###################### depth estimation
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas.eval()
        self.transform = self.midas_transforms.default_transform


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')
        d = self.return_depth(np.asarray(x))


        x = self.img_transform(x)
        y = self.gt_transform(y)
        d = self.gt_transform(d)


        return x , y




    def return_depth(self, img):
        img = cv2.resize(img, (224, 224))
        # convert color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # run midas model
        input_batch = self.transform(img)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        # convert output to numpy array
        output = prediction.cpu().numpy()

        # rescale output for simple depth extraction
        min = np.min(output)
        max = np.max(output)
        output2 = 255.99 * (output - min) / (max - min)
        output2 = output2.astype(int)
        output2 = np.stack((output2,) * 3, axis=-1)
        depth = output2.astype(np.uint8)
        pil_depth = Image.fromarray(depth)
        return pil_depth



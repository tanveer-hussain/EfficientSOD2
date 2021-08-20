from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn

from dpt.transforms import Resize, NormalizeImage, PrepareForNet

class DatasetLoader(Dataset):


    def __init__(self, dir, d_type):

        self.x_path = os.path.join(dir, str(d_type), 'Images')

        self.y_path = os.path.join(dir, str(d_type), 'Labels')

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)

        self.length = len(self.X)

        # self.transform = Compose(
        #     [
        #         Resize(
        #             net_w,
        #             net_h,
        #             resize_target=None,
        #             keep_aspect_ratio=True,
        #             ensure_multiple_of=32,
        #             resize_method="minimal",
        #             image_interpolation_method=cv2.INTER_CUBIC,
        #         ),
        #         NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #         PrepareForNet(),
        #     ]
        # )


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])

        x = cv2.imread(x_full_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255.0
        y = cv2.imread(y_full_path,0)

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), PrepareForNet()])

        # x = self.transform({"image": x})["image"]
        # y = self.transform({"image": y})["image"]
        x = transform(x)
        y = transform(y)


        return x , y

import os
from PIL import Image
import cv2
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.depths = sorted(self.depths)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])

        ###################### depth estimation
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas.eval()
        self.transform = self.midas_transforms.default_transform

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

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.return_depth(np.asarray(image))
        # depth = self.rgb_loader(self.depths[index])
        gray = self.binary_loader(self.grays[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        depth = self.depth_transform(depth)
        gray = self.gray_transform(gray)
        # img_names = self.images[index]
        return image, gt, depth, gray, index

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.depths)
        assert len(self.images) == len(self.grays)
        images = []
        gts = []
        depths = []
        grays = []
        for img_path, gt_path, depth_path, gray_path in zip(self.images, self.gts, self.depths, self.grays):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            gray = Image.open(gray_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                grays.append(gray_path)
        self.images = images
        self.gts = gts
        self.depths = depths
        self.grays = grays

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, gray_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root, gray_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, dataset.size

# def update_data_loader(new_z, dataset):
#
#     dataset = SalObjDataset(image_root, gt_root, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader

class test_dataset:
    def __init__(self, image_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        HH = 224
        WW = 224
        image = self.transform(image).unsqueeze(0)
        depth = self.depth_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, depth, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

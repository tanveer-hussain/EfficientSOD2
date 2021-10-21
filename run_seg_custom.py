"""Compute segmentation maps for images in the input folder.
"""
import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F

import util.io

from torchvision.transforms import Compose
from dpt.models_custom import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


# load network
net_w = net_h = 224
model_path = "weights/dpt_hybrid-ade20k-53898607.pt"
model = DPTSegmentationModel(
    150,
    path=model_path,
    backbone="vitb_rn50_384",
)
transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ]
)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimize = True
if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)



print("start processing")

img_name = 'input/150.jpg'
img = cv2.imread(img_name)
img = cv2.resize(img, (net_w,net_h))
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
# img_input = transform({"image": img})["image"]
import numpy as np
img_input = np.transpose(img, (2, 0, 1))
img_input = np.ascontiguousarray(img_input).astype(np.float32)
# compute
with torch.no_grad():
    sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
    if optimize == True and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    x1, x2, x3, x4 = model.forward(sample)
    # out = model.forward(sample)
    #
    # prediction = torch.nn.functional.interpolate(
    #     out, size=img.shape[:2], mode="bicubic", align_corners=False
    # )
    # prediction = torch.argmax(prediction, dim=1) + 1
    # prediction = prediction.squeeze().cpu().numpy()
    # filename = "output_semseg/150"
    # util.io.write_segm_img(filename, img, prediction, alpha=0.5)

    # print (x4.shape, x3.shape, x2.shape, x1.shape)

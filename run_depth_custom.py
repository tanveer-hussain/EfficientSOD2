"""Compute segmentation maps for images in the input folder.
"""
import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F
from dpt.models_custom import DPTSegmentationModel, DPTDepthModel


# load network
net_w = net_h = 224
model_path = "weights/dpt_hybrid-midas-501f0c75.pt"
model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimize = True
if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)

from util import io

print("start processing")

img_name = '1.jpg'
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

    out, x1, x2, x3, x4 = model.forward(sample)
    # prediction = (
    #     torch.nn.functional.interpolate(
    #         out.unsqueeze(1),
    #         size=img.shape[:2],
    #         mode="bicubic",
    #         align_corners=False,
    #     )
    #         .squeeze()
    #         .cpu()
    #         .numpy()
    # )

    # filename = "150"
    # bits = 2
    # depth_min = prediction.min()
    # depth_max = prediction.max()
    #
    # max_val = (2 ** (8 * bits)) - 1
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    # else:
    #     out = np.zeros(prediction.shape, dtype=prediction.dtype)
    #
    # if bits == 1:
    #     cv2.imwrite(filename + ".png", out.astype("uint8"))
    # elif bits == 2:
    #     cv2.imwrite(filename + ".png", out.astype("uint16"))
    #
    # print (x4.shape, "\n", x3.shape, "\n",  x2.shape, "\n",  x1.shape)


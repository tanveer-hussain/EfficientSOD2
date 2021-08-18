import torch
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import pdb, os, argparse
from datetime import datetime
from ResNet_models import Generator
from data import get_loader
from utils import adjust_lr
from scipy import misc
from utils import l2_regularisation
import smoothness
import imageio


# DPT
import cv2
import util.io
from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

net_w = net_h = 28
model_type = 'dpt_hybrid'
default_models = {
        "dpt_large": "weights/dpt_large-ade20k-b12dca68.pt",
        "dpt_hybrid": "weights/dpt_hybrid-ade20k-53898607.pt",
    }
model_weights = default_models[model_type]
dpt_model = DPTSegmentationModel(
            150,
            path=None,
            backbone="vitl16_384",
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
dpt_model.eval()
dpt_model.to(device)


img = util.io.read_image('150.jpg')
img = transform({"image": img})["image"]
img_tensor = torch.from_numpy(img).to(device).unsqueeze(0)
img_tensor = img_tensor.to(memory_format=torch.channels_last)
# img_tensor = img_tensor.half()
out = dpt_model.forward(img_tensor)

print ('tinue')

#Set runtime to GPU
print ("import libraries")
#importing libraries
import tifffile as ti
import random
import numpy as np
import torch
#load midas model from torch hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()
import cv2
import urllib.request
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
import numpy as np
import cv2
import os

use_large_model = True

if use_large_model:
    midas = torch.hub.load("intel-isl/MiDaS", "dpt_large")
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# if use_large_model:
#     midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
# else:
#     midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform

# create output folders for 8-bit simple output and 32-bit fully 3d vector output
try:
  os.makedirs('out8tif')
except:
  print('Folder not created')
pp=1
# for each filename do the depth extraction and save it
for pp in range (pp>0):
  # read file
  img = cv2.imread('0001.jpg')
  img=cv2.resize(img, (400,400))
  # convert color space from BGR to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # run midas model
  input_batch = transform(img).to(device)
  with torch.no_grad():
    prediction = midas(input_batch)

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
  output2 = 255.99*(output-min)/(max-min)
  output2 = output2.astype(int)
  output2 = np.stack((output2,)*3, axis=-1)
  img1=output2.astype(np.uint8)
  # save simple output together with blurred versions
  cv2.imwrite('out8tif/1.jpg', img1)
  
cv2.imshow('sample image',img1)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()

def return_depth():
    pass











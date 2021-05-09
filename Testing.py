import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import cv2
import timeit
import csv


# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##################################################################################
# Directories and models names to be changed
dataset_path = r'D:\My Research\Datasets\Saliency Detection\S-SOD\S-SOD'
base_dir = r"C:\Users\user02\Documents\GitHub\EfficientSOD2"
output_dir = base_dir + r"\SIP\\"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_path = os.path.join(base_dir, 'TrainedModels\\SIP_DDNet_Model_200.pt')
##################################################################################

model = torch.load(model_path)
model.to(device)
model.eval()
kernel = np.ones((5,5), np.uint8)

def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = x.to(device)

    x = torch.unsqueeze(x, 0)
    return x

def predictions(img,d):
    x = preprocess_image(img)
    d = preprocess_image(d)
    start_time = timeit.default_timer()
    output = model(x,d)
    # output0 = output[0]
    # output1 = output[1]
    output = torch.squeeze(output, 0)
    # output1 = torch.squeeze(output1, 0)

    # output0 = output0.detach().cpu().numpy()
    # output0 = output0.dot(255)
    # output0 *= output0.max()/255.0

    output = output.detach().cpu().numpy()
    output = output.dot(255.0)
    # output1 *= output1.max()/255.0

    # output = cv2.bitwise_and(output0,output1)
    # ret,output = cv2.threshold(output,100,255,cv2.THRESH_BINARY)
    # print (max(output))
    # output = cv2.erode(output, kernel, iterations=2)
    # output = cv2.dilate(output, kernel, iterations=1)
    return output
# def testing_code_dir(input_dir, output_dir):
#
#     val_base_path_images = os.listdir(input_dir)
#     for single_image in val_base_path_images:
#         full_path = input_dir + single_image
#         print (full_path)
#         depth_path = r'D:\My Research\Datasets\Saliency Detection\S-SOD\S-SOD\Depth\\' + single_image[0:(len(single_image) - 3)] + "png"
#
#         img = Image.open(full_path).convert("RGB")
#         depth = Image.open(depth_path).convert("L")
#
#         print (f'x > {full_path}, y > {depth_path}')
#
#         output = predictions(img,depth)
#         output = np.transpose(output, (1, 2, 0))
#         # cv2.imshow('', output)
#         # cv2.waitKey(10)
#
#         output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
#         cv2.imwrite(output_path, output)
#         print("Reading: %s\n writing: %s " % (full_path, output_path))
#
# # # testing code SIP
# input_dir = r'D:\My Research\Datasets\Saliency Detection\S-SOD\S-SOD\Images\\'
# output_dir = r'C:\Users\user02\Documents\GitHub\EfficientSOD2\S-SOD\\'
# testing_code_dir(input_dir,output_dir)

# full_path = D:\My Research\Datasets\Saliency Detection\S-SOD\S-SOD\Images\\
# img = Image.open(full_path).convert("RGB")
# depth = Image.open(depth_path).convert("L")
# with open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Test.csv", 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         counter += 1
#         x_image = dataset_path + "\Images\\" + str(row[0])
#         img = Image.open(x_image)
#         output = predictions(img)
#         output = np.transpose(output, (1, 2, 0))
#         output_path = output_dir + str(row[0])[0:(len(str(row[0])) - 3)] + "png"
#         print("writing: %s " % (output_path))
#         cv2.imwrite(output_path, output)

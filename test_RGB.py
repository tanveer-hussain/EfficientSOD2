from multiprocessing.dummy import freeze_support
import torch
import numpy as np
import cv2
import os
from data import RetreiveTestData, TestDatasetLoader
from torch.utils.data import DataLoader
from EvaluateSOD import Evaluator
device = torch.device('cuda' if torch.cuda.is_available else "cpu")
import shutil

class ModelTesting():
    def __init__(self, model, test_loader, output_root, gt_root, dataset_name):
        self.dataset_name = dataset_name
        self.model = model
        self.model.eval()
        self.loader = test_loader
        self.output_path = output_root
        self.gt_root = gt_root + "/Labels"
        self.prediction()
        self.evaluate()

    def prediction(self):
        for iter, (X, _, name) in enumerate(self.loader):
            X = X.to(device)
            pred = self.model.forward(X, training=False)
            output = torch.squeeze(pred, 0)
            output = output.detach().cpu().numpy()
            output = output.dot(255)
            output *= output.max() / 255.0
            output = np.transpose(output, (1, 2, 0))
            image_name , _ = name[0].split('.')
            output_path = self.output_path + image_name + '.png'
            
            None if os.path.exists("results/gt/") else os.mkdir("results/gt/")
            gt_output_path = "results/gt/" + self.dataset_name
            gt_output_path if os.path.exists(gt_output_path) else os.mkdir(gt_output_path)
            
            gt_complete_path = self.gt_root + "/" + image_name + '.png'
            shutil.copy(gt_complete_path, gt_output_path)
            
            print ("Saving Image at.. ", output_path, ", and copying", gt_complete_path, ", to", gt_output_path)
            cv2.imwrite(output_path, output)
    def evaluate(self):
        print (self.output_path, self.gt_root)
        eval_data = TestDatasetLoader(self.output_path, self.gt_root)
        eval_loader = DataLoader(eval_data)

        eval = Evaluator(eval_loader)
        mae , fmeasure, emeasure, smeasure = eval.execute()
        logfile = 'results/EvaluationResults_RGB.txt'
        with open(logfile, 'a+') as f:
            f.write(self.dataset_name  + "\tMAE: " + str(mae) + ", FMeasure: " + str(fmeasure) + ", EMeasure: " + str(emeasure) + ", SMeasure: " + str(fmeasure) + "\n")



# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##################################################################################
# Directories and models names to be changed
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=32, latent_dim=3)
resswin.to(device)
model_path = r'models\\DUT-RGBDRGBD_D_25_Pyramid.pth'
##################################################################################

resswin.load_state_dict(torch.load(model_path))
resswin.to(device)
resswin.eval()
import torchvision.transforms as T

def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = x.to(device)

    x = torch.unsqueeze(x, 0)
    return x

def predictions(img,d):
    x = preprocess_image(img)
    d = preprocess_image(d)
    output = resswin(x,d)
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
from PIL import Image
def testing_code_dir(input_dir, output_dir):

    val_base_path_images = os.listdir(input_dir)
    for single_image in val_base_path_images:
        image_path = input_dir + "/Images\\"  + single_image
        full_path = input_dir + single_image
        print (full_path)
        depth_path = input_dir + "/Depth\\" + single_image[0:(len(single_image) - 3)] + "png"

        img = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")

        print (f'x > {full_path}, y > {depth_path}')

        output = predictions(img,depth)
        output = np.transpose(output, (1, 2, 0))
        # cv2.imshow('', output)
        # cv2.waitKey(10)

        output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
        cv2.imwrite(output_path, output)
        print("Reading: %s\n writing: %s " % (full_path, output_path))
#
# # # testing code SIP
input_dir = r'D:\My Research\Datasets\Saliency Detection\RGBD/DUT-RGBD/Test\\'
output_dir = r'C:\Users\user02\Documents\GitHub\EfficientSOD2\DUT-Testing\\'
testing_code_dir(input_dir,output_dir)

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



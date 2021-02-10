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
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGB\Pascal-S'
base_dir = r"C:\Users\user02\Documents\GitHub\EfficientSOD2"
output_dir = base_dir + r"\PASCAL\\"
model_path = os.path.join(base_dir, 'TrainedModels\\PASCAL_DDNet_500Model.pt')
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
    
def predictions(img):
    x = preprocess_image(img)
    start_time = timeit.default_timer()
    output = model(x)
    output = torch.squeeze(output, 0)
    output = output.detach().cpu().numpy()
    output = output.dot(255)
    output *= output.max()/255.0
    # print (max(output))
    # output = cv2.erode(output, kernel, iterations=2)
    # output = cv2.dilate(output, kernel, iterations=1)
    return output
counter = 1
with open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Test.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        counter += 1
        x_image = dataset_path + "\Images\\" + str(row[0])
        img = Image.open(x_image)
        output = predictions(img)
        output = np.transpose(output, (1, 2, 0))
        output_path = output_dir + str(row[0])[0:(len(str(row[0])) - 3)] + "png"
        print("writing: %s " % (output_path))
        cv2.imwrite(output_path, output)
print (counter)
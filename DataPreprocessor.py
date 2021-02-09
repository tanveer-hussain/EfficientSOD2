import os
import math
from torch.utils.data import random_split
import numpy as np
from sklearn.model_selection import train_test_split

def main():

  

    dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGB\Pascal-S\Images'
    d_type = ['Train', 'Test']
    counter = 0
    all_data = []

    for single_image in os.listdir(dataset_path):

        all_data.append(single_image)


    X_train, X_test = train_test_split(all_data, test_size=0.2, random_state=42)

    

    for single_image in X_test:

        with open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Test.lst", "a") as outfile:
            outfile.write(str(single_image) + "\n")

    for single_image in X_train:

        with open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Train.lst", "a") as outfile:
            outfile.write(str(single_image) + "\n")

    outfile = open(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\Pascal-S_Train.lst", "r")
    outdata = outfile.read().split("\n")
    outfile.close()

        

if __name__ == '__main__':
    main()

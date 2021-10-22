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
            image_name = image_name + '.png'

            gt_path = "results/gt/" if os.path.exists("results/gt/") else os.mkdir("results/gt/")
            gt_path = gt_path + self.dataset_name if os.path.exists(gt_path + self.dataset_name) else os.mkdir(gt_path + self.dataset_name)
            # gt_path =
            shutil.copy(os.path.join(self.gt_root, image_name), gt_path)
            output_path = self.output_path + image_name
            print ("Saving Image at.. ", output_path, ", Copying from ", gt_path, ", to ", gt_path)
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







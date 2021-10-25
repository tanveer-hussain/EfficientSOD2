from multiprocessing.dummy import freeze_support
import torch
import numpy as np
import cv2
import os
from data import RetreiveTestData, TestDatasetLoader
from torch.utils.data import DataLoader
from EvaluateSOD import Evaluator

class ModelTesting():
    def __init__(self, model, test_loader, output_root):
        self.model = model
        self.loader = test_loader
        self.output_path = output_root
        self.prediction()

    def prediction(self):
        for iter, (X, name) in enumerate(self.loader):
            X = X.to(device)
            pred = self.model.forward(X, training=False)
            output = torch.squeeze(pred, 0)
            output = output.detach().cpu().numpy()
            output = output.dot(255)
            output *= output.max() / 255.0
            output = np.transpose(output, (1, 2, 0))
            name, _ = name.split('.')
            output_path = self.output_path + name + '.png'
            print ("Saving Image at.. ", output_path)
            cv2.imwrite(output_path, output)

device = torch.device('cuda' if torch.cuda.is_available else "cpu")
latent_dim=3
feat_channel=32
test_datasets = ['DUT-RGBD', "NJU2K", "NLPR", 'SIP']
# test_datasets = ['DUT-RGBD', "NJU2K"]
dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/'
# dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
epoch = 40
from ResSwin import ResSwinModel
resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
resswin.to(device)


class ModelTesting():
    def __init__(self, model, test_loader, output_root, gt_root, dataset_name):
        self.dataset_name = dataset_name
        self.model = model
        self.model.eval()
        self.loader = test_loader
        self.output_path = output_root
        self.gt_root = gt_root + "/Test/Labels"
        self.prediction()
        self.evaluate()

    def prediction(self):
        for iter, (X, depth, _, name) in enumerate(self.loader):
            X = X.to(device)
            depth = depth.to(device)
            pred = self.model.forward(X, depth, training=False)
            output = torch.squeeze(pred, 0)
            output = output.detach().cpu().numpy()
            output = output.dot(255)
            output *= output.max() / 255.0
            output = np.transpose(output, (1, 2, 0))
            image_name , _ = name[0].split('.')
            output_path = self.output_path + image_name + '.png'
            print ("Saving Image at.. ", output_path)
            cv2.imwrite(output_path, output)
    def evaluate(self):
        print (self.output_path, self.gt_root)
        eval_data = TestDatasetLoader(self.output_path, self.gt_root)
        eval_loader = DataLoader(eval_data)

        eval = Evaluator(eval_loader)
        mae, fmeasure, emeasure, smeasure = eval.execute()
        logfile = 'results/EvaluationResults_RGBD.txt'
        with open(logfile, 'a+') as f:
            f.write(self.dataset_name + "\tMAE: " + str(mae) + ", FMeasure: " + str(fmeasure) + ", EMeasure: " + str(
                emeasure) + ", SMeasure: " + str(fmeasure) + "\n")
        print ("Testing done")

pred_edge_kk *= 255.0
pred_edge_kk = pred_edge_kk.astype(np.uint8)
#
#

#
#     # for evaluating results
#     predictions_root = 'results/' + dataset + '/'
#     # predictions_root if os.path.exists(predictions_root) else os.mkdir(predictions_root)
#     gt_root = r"D:\My Research\Datasets\Saliency Detection\RGBD\\" + dataset + "/Test/Labels\\"
#     eval_data = TestDatasetLoader(predictions_root, gt_root)
#     eval_loader = DataLoader(eval_data)
#
#     eval = Evaluator(eval_loader)
#     mae , fmeasure, emeasure, smeasure = eval.execute()
#     logfile = 'results/EvaluationResults.txt'
#     with open(logfile, 'a+') as f:
#         f.write(dataset + "\tMAE: " + str(mae) + ", FMeasure: " + str(fmeasure) + ", EMeasure: " + str(emeasure) + ", SMeasure: " + str(fmeasure) + "\n")
#
#
#



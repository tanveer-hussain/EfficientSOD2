from multiprocessing.dummy import freeze_support
import torch
import numpy as np
import cv2
import os
from data import RetreiveTestData, TestDatasetLoader
from torch.utils.data import DataLoader
from EvaluateSOD import Evaluator
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available else "cpu")

# device = torch.device('cuda' if torch.cuda.is_available else "cpu")
# latent_dim=3
# feat_channel=32
# test_datasets = ['DUT-RGBD', "NJU2K", "NLPR", 'SIP']
# # test_datasets = ['DUT-RGBD', "NJU2K"]
# dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD/'
# # dataset_path = r'/media/tinu/새 볼륨/My Research/Datasets/Saliency Detection/RGBD/' + dataset_name + '/Test'
# epoch = 40
# from ResSwin import ResSwinModel
# import imageio
# resswin = ResSwinModel(channel=feat_channel, latent_dim=latent_dim)
# resswin.to(device)


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
        for iter, (X, depth, width, height, _, name) in enumerate(self.loader):
            X = X.to(device)
            depth = depth.to(device)

            pred = self.model.forward(X, depth)
            output = F.upsample(pred, size=[width, height], mode='bilinear', align_corners=False)
            # output = pred.sigmoid().data.cpu().numpy().squeeze()
            output = output.detach().cpu().numpy()
            output = output.dot(255)
            output *= output.max() / 255.0
            output = np.transpose(output, (1, 2, 0))
            image_name , _ = name[0].split('.')
            output_path = self.output_path + image_name + '.png'
            print ("Saving Image at.. ", output_path)
            cv2.imwrite(output_path, output*255)
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
# import cv2
# def visualize_uncertainty_prior_init(var_map,nature):
#
#     for kk in range(var_map.shape[0]):
#         pred_edge_kk = var_map[kk, :, :, :]
#         pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
#         # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
#         pred_edge_kk *= 255.0
#         pred_edge_kk = pred_edge_kk.astype(np.uint8)
#         # print('proir_edge_kk shape', pred_edge_kk.shape)
#         save_path = './confirm/'
#         if nature == "i":
#             name = '{:02d}_image.png'.format(kk)
#         else:
#             name = '{:02d}_depth.png'.format(kk)
#         pred_edge_kk = np.transpose(pred_edge_kk, (1, 2, 0))
#         cv2.imwrite(save_path + name, pred_edge_kk)


#
# datasets = ["DUT-RGBD", "NLPR", 'NJU2K', 'SIP']
# dataset_name = datasets[0]
# dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD\\' + dataset_name
#
#
# from ResSwin import ResSwinModel
# resswin = ResSwinModel(channel=32, latent_dim=3)
# resswin.cuda()
# msg = resswin.load_state_dict(torch.load(r"C:\Users\user02\Documents\GitHub\EfficientSOD2\models/DUT-RGBDRGBD_D_1_Pyramid.pth"))
# print ('Weights loaded', msg)
# resswin.eval()
#
# from data import TrainDatasetLoader
# save_path = r'C:\Users\user02\Documents\GitHub\EfficientSOD2\results\DUT-RGBD'
#
# print (dataset_path)
# test_loader = TrainDatasetLoader(dataset_path, 'Test')
# for iter, (X, depth, _, name) in enumerate(test_loader):
#
#     X = X.unsqueeze(0)
#     depth = depth.unsqueeze(0)
#
#
#     image = X.cuda()
#     depth = depth.cuda()
#
#     visualize_uncertainty_prior_init(X, 'i')
#     visualize_uncertainty_prior_init(depth, 'd')
#
#     import timeit
#
#     start_time = timeit.default_timer()
#     generator_pred = resswin.forward(image, depth)
#     #print('Single prediction time consumed >> , ', timeit.default_timer() - start_time, ' seconds')
#     print (generator_pred.shape)
#     res = generator_pred
#     # res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
#     res = res.sigmoid().data.cpu().numpy().squeeze()
#     # name = name[:-3]
#     output_path = os.path.join(save_path,name)
#     # cv2.imshow('',res)
#     # cv2.waitKey(0)
#     print(output_path)
#
#     cv2.imwrite(output_path, res*255)

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



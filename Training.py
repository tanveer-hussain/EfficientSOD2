import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import RGBDNetwork
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
from torchsummary import summary
import  os
import cv2
import sys
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    print('__Number of CUDA Devices:', cuda.device_count(), ', active device ID:', cuda.current_device())
    print ('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())
    model = RGBDNetwork.DenseNetBackbone()
    base_dir = r"C:\Users\user02\Documents\GitHub\EfficientSOD2\\"

    cudnn.benchmark = True
    model.to(device)
    print (count_parameters(model))
    # print (model)
    # summary(model,(3, 224,224),(1, 224,224))

    base_lr = 0.0001
    epochs = 200
    # x = torch.cat((x,d), 1)
    weight_decay = 1e-3
    
    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    # criterion = KLDLoss().to(device)
    
    print('Model on GPU: ', next(model.parameters()).is_cuda)

    dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGBD\SIP'

    print ('Loading training data...')
    X = []
    Y = []
    D = []

    d_type = ['Train', 'Test']

    train_data = DatasetLoader(dataset_path, d_type[0])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1, drop_last=True)
    # total_training_loss = []
    # total_validation_loss = []
    # print ('Training..')
    for epoch in range(0, epochs):

        model.train()
        training_loss = 0

        for X , Y, D in train_loader:
            optimizerr.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            D = D.to(device)
            RGOutput = model(X, D)

            # temp = RGOutput[0].detach().cpu()
            # temp = temp.numpy()
            # temp = np.transpose(temp, (1, 2, 0))
            # cv2.imshow('', temp)
            # cv2.waitKey(1)

            loss = criterion(RGOutput, Y)
            # Depthloss = criterion(DepthOutput, Y)

            # loss = RGBloss + Depthloss
            loss.backward()
            optimizerr.step()
            training_loss += loss.item()


    
        training_loss /= len(train_loader)
    #     validation_loss /= len(validation_loader)
        if epoch%2 == 0:
            print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {training_loss}')
            # print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {training_loss}.. Validation Loss: {validation_loss}')

    
    torch.save(model, os.path.join(base_dir, 'TrainedModels\\SIP_DDNet_Model_200.pt'))
    # torch.save(model.state_dict(), os.path.join(base_dir,'TrainedModels\\PASCAL_DDNet_500Weights.pt'))

if __name__ == '__main__':
    main()

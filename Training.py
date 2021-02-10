import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import BaseNetwork_3
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
import  os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, opt, crit, train_loader, epoch):
    model.train()
    for i, (X, Y) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)
    
        output = model(X)

        loss = crit(output, Y)
    
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

def main():

    print('__Number of CUDA Devices:', cuda.device_count(), ', active device ID:', cuda.current_device())
    print ('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())
    model = BaseNetwork_3.DenseNetBackbone()
    base_dir = r"C:\Users\user02\Documents\GitHub\EfficientSOD2\\"

    cudnn.benchmark = True
    model.to(device)

    base_lr = 0.0001
    epochs = 500
    weight_decay = 1e-3
    
    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    
    print('Model on GPU: ', next(model.parameters()).is_cuda)

    dataset_path = r'D:\My Research\Datasets\Saliency Detection\RGB\Pascal-S'

    print ('Loading training data...')
    X = []
    Y = []

    train_data = DatasetLoader(dataset_path, X, Y)
    
    train_loader = DataLoader(train_data, batch_size=48, shuffle=True, num_workers=1, drop_last=True)
    # total_training_loss = []
    # total_validation_loss = []
    # print ('Training..')
    for epoch in range(0, epochs):

        model.train()
        training_loss = 0

        for X , Y in train_loader:
            optimizerr.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)

            loss = criterion(output, Y)
            loss.backward()
            optimizerr.step()
            training_loss += loss.item()

    #     model.eval()
    #     validation_loss = 0

    #     with torch.no_grad():
    #         for X , Y in validation_loader:

    #             X = X.to(device)
    #             Y = Y.to(device)
    #             output = model(X)
    #             loss = criterion(output, Y)

    #             validation_loss += loss.item()
    #             total_validation_loss.append(total_validation_loss)

        training_loss /= len(train_loader)
    #     validation_loss /= len(validation_loader)
        if epoch%4 == 0:
            print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {training_loss}')
            # print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {training_loss}.. Validation Loss: {validation_loss}')

    # plt.plot(total_training_loss , marker='*', label='Training Loss', color='darkorange')
    # plt.plot(total_validation_loss , marker='+', label='Validation Loss', color='black')
    # plt.grid(True)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend(loc=1)
    # # plt.savefig('TrainingLoss.png')
    # plt.show()
    
    torch.save(model, os.path.join(base_dir, 'TrainedModels\\PASCAL_DDNet_500Model.pt'))
    torch.save(model.state_dict(), os.path.join(base_dir,'TrainedModels\\PASCAL_DDNet_500Weights.pt'))

if __name__ == '__main__':
    main()

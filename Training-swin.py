import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import BaseNetwork_4
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from ModelNetworks import model

model = model.ResNet50(num_classes=1000, resolution=(224, 224))
x = torch.randn([2, 3, 225, 224])
print(model)



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    model = 1
    print (model)


if __name__ == '__main__':
    main()

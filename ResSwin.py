import torch

from ResNet_models_UCNet import Generator
device = torch.device('cuda' if torch.cuda.is_available else "cpud")
generator = Generator(32, 3)
generator.cuda()

class ResSwin():
    def __init__(self):

x = torch.randn((8, 3, 224, 224)).to(device)
depth = torch.randn((8, 3, 224, 224)).to(device)
gt = torch.randn((8, 1, 224, 224)).to(device)
sal_init, depth_pred = generator(x,depth, gt)
print(sal_init.shape, " > ", depth_pred.shape)
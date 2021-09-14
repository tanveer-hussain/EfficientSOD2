from ResNet_models_UCNet import Generator

model = Saliency_feat_encoder(32,3).to(device)

x = torch.randn((8, 3, 224, 224)).to(device)
depth = torch.randn((8, 3, 224, 224)).to(device)
gt = torch.randn((8, 1, 224, 224)).to(device)
sal_init, depth_pred = model(x,depth, gt)
print(sal_init.shape, " > ", depth_pred.shape)
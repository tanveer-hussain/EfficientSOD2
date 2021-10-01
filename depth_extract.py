from AE_model_unet import *
import os
import torch.backends.cudnn as cudnn

from imageio import imread
from PIL import Image
import scipy.misc
from torch.autograd import Variable
import collections
import argparse
import cv2

parser = argparse.ArgumentParser(description='Pretrained Depth AutoEncoder',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_dir', type=str, default = "C:/Users/IMLab/Desktop/tai_dataset/depth/GDN-Pytorch-master/model/KITTI_models/GDN_RtoD_pretrained.pkl")                                 
parser.add_argument('--gpu_num', type=str, default = "0")
args = parser.parse_args()

def load_as_float(path):
    return imread(path).astype(np.float32)

class Resize(object):
    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation
    def __call__(self, img,size, img_type = 'rgb'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        if img_type == 'rgb':
            if img.ndim == 3:
                return scipy.misc.imresize(img, size, self.interpolation)
            elif img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation)
                img_tmp = np.zeros((img.shape[0], img.shape[1],1),dtype=np.float32)
                img_tmp[:,:,0] = img[:,:]
                img = img_tmp
                return img
        elif img_type == 'depth':
            if img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation, 'F')
            elif img.ndim == 3:
                img = scipy.misc.imresize(img[:,:,0], size, self.interpolation, 'F')
            img_tmp = np.zeros((img.shape[0], img.shape[1],1),dtype=np.float32)
            img_tmp[:,:,0] = img[:,:]
            img = img_tmp
            return img
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
upsampling = nn.functional.interpolate
resize = Resize()

ae = AutoEncoder()
ae = ae.cuda()
ae = nn.DataParallel(ae)
ae.load_state_dict(torch.load("GDN_RtoD_pretrained.pkl"))
ae = ae.eval()

cudnn.benchmark = True
torch.cuda.synchronize()
sum = 0.0
images = cv2.imread("1.jpg")


sample = []
#filenames = []
org_H_list = []
org_W_list = []
#for filename in img_list:
img = images#load_as_float(filename)
org_H_list.append(img.shape[0])
org_W_list.append(img.shape[1])

img = cv2.resize(img,(128,416))
img = img.transpose(2,0,1)
img = torch.tensor(img,dtype=torch.float32)
img = img.unsqueeze(0)
img = img/255
img = (img-0.5)/0.5
#img = upsampling(img, (128,416),mode='bilinear', align_corners = False)
img = Variable(img)
sample.append(img)
#filenames.append(filename.split('/')[-1])

print("sample len: ",len(sample))

i=0
result_dir = "/home/tinu/PycharmProjects/EfficientSOD2"
k=0
t=0
img_ = None
for tens in sample:
    #print (tens)
    filename = images
    org_H = org_H_list[i]
    org_W = org_W_list[i]
    torch.cuda.synchronize()
    #print(tens.size())
    img = ae(tens,istrain=False)
    if i>0:
        sum += tmp
    #print(img.size())
    img = upsampling(img, (128,416),mode='bilinear', align_corners = False)
    img = img[0].cpu().detach().numpy()
    img = img.astype(np.uint8)
    img_tmp = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
    img_tmp[:, :, 0] = img[:, :]
    img = img_tmp
    cv2.imshow('sample image', img)
    cv2.waitKey(5000)
    #print(img.shape)
    if img.shape[0] == 3:
        img_ = np.empty([128,416,3])
        img_[:,:,0] = img[0,:,:]
        img_[:,:,1] = img[1,:,:]
        img_[:,:,2] = img[2,:,:]
    elif img.shape[0] == 1:
        img_ = np.empty([128,416])
        img_[:,:] = img[0,:,:]
    img_ = cv2.resize(img_, (org_H, org_W))
    print (img_.shape)
    # if img_.shape[2] == 1:
    img_ = img_[:,:,0]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    #print(result_dir)
    #print(img_.shape)
    print (img_.shape)
    img1=img_.astype(np.uint8)
    #cv2.imwrite('lena_bgr_cv.jpg', img1)
    #imageio.imsave('Depth.jpg', img1)
cv2.imshow('sample image',img1)
cv2.waitKey(0)


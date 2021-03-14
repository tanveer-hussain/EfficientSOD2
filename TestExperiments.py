import cv2
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

base_path = "D:\My Research\Datasets\Saliency Detection\RGBD\SIPALL\Depth"


img = Image.open(os.path.join(base_path,'3.png')).convert('RGB') #(os.path.join(base_path,'226.png'), 0)
img = np.array(img)

# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (50,50,450,290)
# cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

edges = cv2.Canny(img,100,200)
# img = Image.fromarray(img)
# # image.show()
cv2.imshow('', edges)
cv2.waitKey(0)
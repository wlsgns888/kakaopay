import os
import json
import cv2
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from config.config import *

class CustomDataset(Dataset):
    def __init__(
        self, 
        files, 
        transforms,
        num_class = 2,
        mode = 'train',
        img_size = 112,
    ) -> None:
        self.mode = mode
        self.files = files
        self.nc = num_class
        self.transforms = transforms
        self.label_dict = label_dict(num_class)
        #self.files = [x if x in img_exes else continue for x in files]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        image_path = self.files[i]
        file_name = image_path.split(os.sep)[-1]
        
        # label {}_{}_{}
        #np_label = np.zeros(self.nc)
        annot = file_name[:5]
        label = self.label_dict[annot]
        
        # image
        org_img = cv2.imread(image_path)
        img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if self.mode == 'train' or self.mode == 'val':
            return {
                'img': img,
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'img': img.expand(1,3,112,112),
                'label': torch.tensor(label, dtype=torch.long),
                'org_img': org_img,
                'file_name': image_path,
            }

if __name__ == '__main__':
    import glob,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
    from utils import *
    data_dir = 'new_train_3'
    img_files = get_img_files(data_dir)
    
    data_ = CustomDataset(img_files,2)
    
    for i in range(len(data_)):
        a = data_.__getitem__(i)
        cv2.imshow('img', a['img'])
        cv2.waitKey(1)
    a = 1
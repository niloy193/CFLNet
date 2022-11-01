
import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Get_Patch
import os
import math

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, cfg, is_patch = True):
        'Initialization'
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.cfg = cfg
            
        self.normalize = transforms.Normalize(cfg['dataset_params']['mean'], cfg['dataset_params']['std'])
        self.is_patch = is_patch
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        im_size = self.cfg['dataset_params']['im_size']
        image = cv2.imread(self.list_IDs[index],1)
        image = cv2.resize(image, (im_size,im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)
    
        mask = cv2.imread(self.labels[index],0)
        mask = cv2.resize(mask, (im_size,im_size), interpolation = cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        # spliced_patch_len = self.cfg['dataset_params']['s_patch_len']
        # pristine_patch_len = self.cfg['dataset_params']['p_patch_len']
        # patch_size = self.cfg['dataset_params']['patch_size']
        # if (self.is_patch == True) and (self.cfg['global_params']['with_con'] == True):
        #     gp = Get_Patch(self.cfg)
        #     ps = torch.empty((spliced_patch_len,mask.shape[0],mask.shape[1]))
        #     for i in range(spliced_patch_len):
        #         m = gp.get_spliced_patch(mask, patch_size, 5)
        #         ps[i] = m
        #     pp = torch.empty((pristine_patch_len,mask.shape[0],mask.shape[1]))
        #     for i in range(pristine_patch_len):
        #         m = gp.get_pristine_patch(mask, patch_size, 5)
        #         pp[i] = m
        #     return image, mask, ps, pp

        return image, mask

def get_file_names(cfg):
    root_dir = cfg['dataset_params']['imd_2020_dir']
    all_files = os.listdir(root_dir)
    train_all_IDs = {}
    mask_all_IDs = {}
    count = 0
    for files in all_files:
        all_file = os.listdir(os.path.join(root_dir,files))
        for file_name in all_file:
            if len(file_name.split('_0.jpg'))==2:
                train_all_IDs[count] = files+'/'+file_name
                mask_all_IDs[count] = files+'/'+file_name.split('_0.jpg')[0]+'_0_mask.png'
                count +=1
    rand = np.random.choice(len(train_all_IDs), math.ceil(len(train_all_IDs)*0.1), replace=False).tolist()

    train_IDs = {}
    mask_IDs = {}
    test_IDs = {}
    mask_test_IDs = {}

    count = 0
    test_count = 0
    for i in range(len(train_all_IDs)):
        if i in rand:
            test_IDs[test_count] = os.path.join(root_dir,train_all_IDs[i])
            mask_test_IDs[test_count] = os.path.join(root_dir,mask_all_IDs[i])
            test_count += 1
        else:
            train_IDs[count] = os.path.join(root_dir,train_all_IDs[i])
            mask_IDs[count] = os.path.join(root_dir,mask_all_IDs[i])
            count += 1
    
    return train_IDs, mask_IDs, test_IDs, mask_test_IDs

class generator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.train_IDs, self.mask_IDs, self.test_IDs, self.mask_test_IDs = get_file_names(cfg)

    def get_train_generator(self):
        
        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': True,
                    'pin_memory':True,
                    'num_workers': 4}
        training_set = Dataset(self.train_IDs, self.mask_IDs, self.cfg, is_patch=True)
        training_generator = data.DataLoader(training_set, **params)

        return training_generator

    def get_val_generator(self):

        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': False,
                    'pin_memory':True,
                    'num_workers': 4}
        val_set = Dataset(self.test_IDs, self.mask_test_IDs, self.cfg, is_patch = False)
        validation_generator = data.DataLoader(val_set, **params)
        return validation_generator


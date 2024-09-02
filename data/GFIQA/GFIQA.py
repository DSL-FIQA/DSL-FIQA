import os
import torch
import numpy as np
import cv2
import pandas as pd
from glob import glob
import json


class GFIQA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, mode, transform):
        super(GFIQA, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.mode = mode
        self.transform = transform

        dis_files_data, score_data = [], []
        file_dir = os.path.join(self.image_path, mode)
        file_list = sorted(glob(os.path.join(file_dir, 'img/*png')))
        with open(os.path.join(file_dir, 'label.json'), 'r') as json_file:
            all_label = json.load(json_file)

        label_dict = {}
        img_name = list(all_label['Image'])
        score = list(all_label['MOS'])     
        for i, name in enumerate(img_name):
            label_dict[name] = score[i]
        score_data = []
        landmark_data = []
        for i in file_list:
            fname = i.split('/')[-1]
            score = label_dict[fname]
            score_data.append(score)
            landmark = np.load(os.path.join(file_dir, 'landmark', i.rsplit('/', 1)[-1][:-4]+'.npy'))
            landmark_data.append(landmark)
            


        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list':file_list, 'score_list': score_data, 'landmark_list': landmark_data}
        

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(d_img_name, cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]
        landmark = self.data_dict['landmark_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score,
            'landmark_org': landmark
        }
        if self.transform:
            sample = self.transform(sample)

        if self.mode == 'test':
            return sample, d_img_name
        else: return sample
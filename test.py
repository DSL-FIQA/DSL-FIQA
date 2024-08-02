import os
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize
from tqdm import tqdm
from models.iqa import IQA, degradation_Encoder, contrastive_loss

from glob import glob
from tqdm import tqdm
import json
import torch
from scipy.stats import spearmanr, pearsonr
from pyhocon import ConfigFactory

import argparse
import pandas as pd
from ptflops import get_model_complexity_info
from thop import profile
import time


parser = argparse.ArgumentParser()
parser.add_argument("--g", type= str, default = '0')
parser.add_argument("--exp", type= str, default = 'exp1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.g



def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, dataset, transform, num_crops=20, num_landmark=500, normalization=True, img_size=224):
        super(Image, self).__init__()

        self.img_name = image_path

        read_path = os.path.join('./dataset', dataset, 'test', 'img', self.img_name)

        self.img = cv2.imread(read_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))
        self.landmark = np.load(read_path.replace('img', 'landmark').replace('.png', '.npy'))

        self.transform = transform
        self.num_landmark = num_landmark
        self.normalization = normalization

        c, h, w = self.img.shape
        new_h = img_size
        new_w = img_size

        self.img_patches = []

        self.not_landmark = []
        other_point = len(self.landmark) + 1
        self.points_patch = []

        for i in range(num_crops):
            self.filtered_landmark = []
            points = []
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

            for idx in range(len(self.landmark)):
                if top <= self.landmark[idx, 1] <= top+new_h and left <= self.landmark[idx, 0] <= left+new_w:
                    self.filtered_landmark.append(idx)
                else:
                    self.not_landmark.append(self.landmark[idx,:])
            if len(self.filtered_landmark) < self.num_landmark:
                res = self.num_landmark - len(self.filtered_landmark)
                points = self.filtered_landmark
                points += [other_point] * res
            else:
                points_idx = np.random.choice(len(self.filtered_landmark), size=self.num_landmark, replace=False)
                points += [self.filtered_landmark[i] for i in points_idx]

            if self.normalization:
                points = np.array([x for x in points])
                array_min, array_max = points.min(), points.max()
                points = ((points - array_min) / (array_max - array_min + 1e-6)) * 2 - 1
                self.points_patch.append(points)

        self.points_patch = np.array(self.points_patch)
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        landmark_patch = self.points_patch[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name, 'landmark_org': landmark_patch}
        if self.transform:
            sample = self.transform(sample)
        return sample




if __name__ == '__main__':
    cpu_num = 20
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)
    
    criterion = torch.nn.MSELoss()
    
    # config file
    json_path = os.path.join('./dataset', args.exp, 'test', 'label.json')
    with open(json_path, 'r') as f:
        gt = json.load(f) 
    file_list = gt['Image']
    score_list = gt['MOS']
    print('Number of testing images: {}'.format(len(file_list)))
    
    losses = []
    all_pred = []
    all_label = []    
    all_name = []
    all_error = []
    f = open("./config.conf")
    conf_text = f.read()
    f.close()
    config = ConfigFactory.parse_string(conf_text)

    ckpt_path_DE = os.path.join('./ckpt', args.exp, 'DE.pt')  
    ckpt_path_net = os.path.join('./ckpt', args.exp, 'IQA.pt')  


    # model defination
    DE = degradation_Encoder()
    DE = DE.cuda()
    DE.load_state_dict(torch.load(ckpt_path_DE), strict=False)

    # model defination
    net = IQA(embed_dim=config["Train.embed_dim"], num_outputs=config["Train.num_outputs"], dim_mlp=config["Train.dim_mlp"],
        patch_size=config["Train.patch_size"], img_size=config["Train.img_size"], window_size=config["Train.window_size"],
        depths=config["Train.depths"], num_heads=config["Train.num_heads"], num_tab=config["Train.num_tab"], scale=config["Train.scale"], freq=config["Train.freq"],
        use_landmark=config["Train.use_landmark"], add_mlp=config["Train.add_mlp"])

    net.load_state_dict(torch.load(ckpt_path_net), strict=False)
    net = net.cuda()

    print('\nModel:{}'.format(args.exp))
    for i, image_path in enumerate(tqdm(file_list)):               
        labels = torch.tensor(score_list[i])
        
        # data load
        Img = Image(image_path=image_path, dataset=args.exp,
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            num_crops=config["Train.num_crops"], num_landmark=config["Train.num_landmark"], normalization=config["Train.normalization"], img_size=config["Train.img_size"])        
        avg_score = 0
        for i in range(config["Train.num_crops"]):
            with torch.no_grad():
                DE.eval()
                net.eval()
                patch_sample = Img.get_patch(i)
                patch = patch_sample['d_img_org'].cuda()
                patch = patch.unsqueeze(0)
                landmark = patch_sample['landmark_org'].cuda()
                landmark = landmark.unsqueeze(0)
                style_q, __ = DE(patch)
                score = net(patch, style_q, landmark)
                avg_score += score
        
        pred = avg_score / config["Train.num_crops"]
        loss = criterion(pred, labels.cuda())
        losses.append(loss.item())

        # save results in one epoch
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        all_pred = np.append(all_pred, pred)
        all_label = np.append(all_label, labels)  
        all_name = np.append(all_name,image_path.split('/')[-1][:-4])
        all_error = np.append(all_error, loss.item())


    # 創建一個 DataFrame
    df = pd.DataFrame({'Name': all_name, 'Predict': all_pred, 'GT': all_label, 'Error': all_error})
    # 將 DataFrame 存為 CSV 檔
    save_dir = os.path.join("./result", args.exp)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'prediction.csv'), index=False)

    rho_s, _ = spearmanr(np.squeeze(all_pred), np.squeeze(all_label))
    rho_p, _ = pearsonr(np.squeeze(all_pred), np.squeeze(all_label))
    print('Rho_s: {}  Rho_p: {}  MSE: {} '.format(rho_s, rho_p, np.mean(losses)))




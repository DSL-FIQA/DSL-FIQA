import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--g", type= str, default = '0')
parser.add_argument("--dataset", type= str, default = 'GFIQA')
parser.add_argument("--is_continue",  action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.g

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import logging
import time
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.iqa import IQA, degradation_Encoder, contrastive_loss
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

from pyhocon import ConfigFactory
from shutil import copyfile
import shutil, errno
import distutils
from utils.inference_process import ToTensor as IToTensor
from utils.inference_process import Normalize as INormalize



def copyanything(root_src_dir, root_dst_dir):
    distutils.dir_util.copy_tree(root_src_dir, root_dst_dir)

def file_backup(expname):
    dir_lis = ["./"]
    os.makedirs(os.path.join("./exp", expname, "recording"), exist_ok=True)
    for dir_name in dir_lis:
        cur_dir = os.path.join("./exp", expname, "recording", dir_name[2:])
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == ".py":
                copyfile(
                    os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                )
    dir_lis = ["./data","./timm","./utils","./models"]

    for dir_name in dir_lis:
        cur_dir = os.path.join("./exp", expname, "recording", dir_name[2:])
        os.makedirs(cur_dir, exist_ok=True)
        copyanything(dir_name, cur_dir)

    copyfile(
        "./config.conf", os.path.join("./exp", expname, "recording", "config.conf")
    )

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logging(path_log, log_file):
    if not os.path.exists(path_log): 
        os.makedirs(path_log)
    filename = os.path.join(path_log, log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff*diff + self.epsilon*self.epsilon))
        return loss


def train_epoch(epoch, net, DE, criterion, optimizer, scheduler, train_loader, ch_loss):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for data in tqdm(train_loader):
        x_d = data['d_img_org'].to("cuda")
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to("cuda")
        landmark = data['landmark_org'].to("cuda")

        style_q_net, __ = DE(x_d)
        pred_d = net(x_d, style_q_net, landmark)

        optimizer.zero_grad()
        loss = ch_loss(torch.squeeze(pred_d), labels)
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)

    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p

def eval_epoch(config, epoch, net, DE, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        DE.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config["Train.num_avg_val"]):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                landmark = data['landmark_org'].cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config, land_mark = landmark)
                style_q_net, __ = DE(x_d)
                pred += net(x_d, style_q_net.detach(), landmark)

            pred /= config["Train.num_avg_val"]
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        return np.mean(losses), rho_s, rho_p



if __name__ == '__main__':
    f = open("./config.conf")
    conf_text = f.read()
    f.close()
    config = ConfigFactory.parse_string(conf_text)

    cpu_num = config["Train.cpu_num"]
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    expname = args.dataset



    if config["Train.seed"]:
        setup_seed(20)

    file_backup(expname)
    log_file = expname + "_iqa.log"


    path_tensorboard = os.path.join("./exp", expname, "tensorboard")
    path_ckpt = os.path.join("./exp", expname, "ckpt")
    path_log = os.path.join("./exp", expname, "log")


    if not os.path.exists(path_ckpt):
        os.makedirs(path_ckpt)
        
    if not os.path.exists(path_tensorboard):
        os.makedirs(path_tensorboard)

    set_logging(path_log, log_file)
    logging.info(config)

    writer = SummaryWriter(path_tensorboard)
    if args.dataset == 'GFIQA':
        print('Loaded GFIQA dataset!')
        from data.GFIQA.GFIQA import GFIQA
        image_path = config["Dataset.GFIQA_path"]
        label_path = config["Dataset.GFIQA_label"]
        Dataset = GFIQA
    elif args.dataset == 'CGFIQA':
        print('Loaded CGFIQA dataset!')
        from data.CGFIQA.CGFIQA import CGFIQA
        image_path = config["Dataset.CGFIQA_path"]
        label_path = config["Dataset.CGFIQA_label"]
        Dataset = CGFIQA        
    else:
        raise ValueError("Custom dataset")



    train_dataset = Dataset(
        image_path=image_path,
        label_path=label_path,
        mode="train",
        transform=transforms.Compose([RandCrop(patch_size=config["Train.crop_size"], num_landmark=config["Train.num_landmark"], normalization=config["Train.normalization"]), 
            Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config["Train.prob_aug"]), ToTensor()]),
    )
    val_dataset = Dataset(
        image_path=image_path,
        label_path=label_path,
        mode="val",
        transform=transforms.Compose([RandCrop(patch_size=config["Train.crop_size"], num_landmark=config["Train.num_landmark"], normalization=config["Train.normalization"]),
            Normalize(0.5, 0.5), ToTensor()]),
    )


    print('number of train scenes: {}'.format(len(train_dataset)))
    print('number of val scenes: {}'.format(len(val_dataset)))


    batch_size = config["Train.batch_size"]
        
    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        num_workers=config["Train.num_workers_train"], drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
        num_workers=config["Train.num_workers_val"], drop_last=True, shuffle=False)
    


    DE = degradation_Encoder()
    DE = nn.DataParallel(DE)
    DE = DE.cuda()   

    print("Load Degradation Pretrained Model")
    path_DE = os.path.join('./ckpt', args.dataset, "DE.pt")
    DE.module.load_state_dict(torch.load(path_DE))
    print("Loaded pre-trained DE weight from {}\n".format(path_DE))
    for name, param in DE.named_parameters():
        param.requires_grad = False


    # model defination
    net = IQA(embed_dim=config["Train.embed_dim"], num_outputs=config["Train.num_outputs"], dim_mlp=config["Train.dim_mlp"],
        patch_size=config["Train.patch_size"], img_size=config["Train.img_size"], window_size=config["Train.window_size"],
        depths=config["Train.depths"], num_heads=config["Train.num_heads"], num_tab=config["Train.num_tab"], scale=config["Train.scale"], freq=config["Train.freq"],
        use_landmark=config["Train.use_landmark"], add_mlp=config["Train.add_mlp"])

    ch_loss = CharbonnierLoss()
    net = nn.DataParallel(net)
    net = net.cuda()     

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), DE.parameters())) / 10 ** 6))
    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

 


    # Train continuously
    if args.is_continue:
        path_DE = os.path.join('./ckpt', args.dataset, "DE.pt")
        path_net = os.path.join('./ckpt', args.dataset, "IQA.pt")
        DE.module.load_state_dict(torch.load(path_DE))
        net.module.load_state_dict(torch.load(path_net))
        print("\nTrain continouesly")
        print("Loaded pre-trained DE weight from {}".format(path_DE))
        print("Loaded pre-trained IQA weight from {}".format(path_net))



    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config["Train.learning_rate"],
        weight_decay=config["Train.weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["Train.T_max"], eta_min=config["Train.eta_min"])


    # train & validation
    losses, scores = [], []
    best_srocc_train = 0.
    best_plcc_train = 0.
    main_score_train = 0.
    best_epoch_train = 0.
    best_rmse_train = 0.

    best_srocc_val = 0.
    best_plcc_val = 0.
    main_score_val = 0.
    best_epoch_val = 0.
    best_rmse_val = 0.


    for epoch in range(0, config["Train.n_epoch_iqa"]):
        start_time = time.time()
        logging.info('Running training epoch {} under exp {}'.format(epoch + 1, expname))
        print("\n\n==============        Epoch: {}        ==============".format(epoch + 1))
        print("Experiemntal Case: {}".format(expname))
        print('Starting Training...')

        loss, rho_s, rho_p = train_epoch(epoch, net, DE, criterion, optimizer, scheduler, train_loader, ch_loss)



        writer.add_scalar("Train_loss", loss, epoch)
        writer.add_scalar("Train_SRCC", rho_s, epoch)
        writer.add_scalar("Train_PLCC", rho_p, epoch)



        if (epoch + 1) % config["Train.val_freq"] == 0:
            print('Starting Validation...')
            logging.info('Starting eval...')
            loss_val, rho_s_val, rho_p_val = eval_epoch(config, epoch, net, DE, criterion, val_loader)
            logging.info('Eval done...')



            logging.info('Validation ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(loss_val, rho_s_val, rho_p_val))
            print('Validation ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(loss_val, rho_s_val, rho_p_val))


            writer.add_scalar("Val_loss", loss_val, epoch)
            writer.add_scalar("Val_SRCC", rho_s_val, epoch)
            writer.add_scalar("Val_PLCC", rho_p_val, epoch)

            if rho_s_val > best_srocc_val:
                best_srocc_val = rho_s_val
                best_plcc_val = rho_p_val
                best_epoch_val = epoch + 1
                best_rmse_val = loss_val

                # save weights
                logging.info('Find better Validation Model: epoch:{}, Model:{}, SRCC:{:.4}, PLCC:{:.4}'.format(best_epoch_val, expname, best_srocc_val, best_plcc_val))
                model_name = "IQA.pt"
                model_save_path = os.path.join(path_ckpt, model_name)
                torch.save(net.module.state_dict(), model_save_path)





        logging.info('Best validation model is epoch:{}, loss:{:.4}, SRCC:{:.4}, PLCC:{:.4}'.format(best_epoch_val, best_rmse_val, best_srocc_val, best_plcc_val))
        print('Best validation model is epoch:{}, loss:{:.4}, SRCC:{:.4}, PLCC:{:.4}'.format(best_epoch_val, best_rmse_val, best_srocc_val, best_plcc_val))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
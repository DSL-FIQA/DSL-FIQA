import torch
import numpy as np
import cv2

def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config["Train.crop_size"])
    left = np.random.randint(0, w - config["Train.crop_size"])
    d_img_org = crop_image(top, left, config["Train.crop_size"], img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None, landmark = None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


def five_point_crop(idx, d_img, config, land_mark):
    new_h = config["Train.crop_size"]
    new_w = config["Train.crop_size"]
    b, c, h, w = d_img.shape
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    d_img_org = crop_image(top, left, config["Train.crop_size"], img=d_img, landmark= land_mark)

    return d_img_org



class RandCrop(object):
    def __init__(self, patch_size, num_landmark = 500, normalization = True):
        self.patch_size = patch_size
        self.num_landmark = num_landmark
        self.normalization = normalization
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        landmark = sample['landmark_org']
        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        filtered_landmark = []
        not_landmark = []
        develop = True
        points = []
        other_point = len(landmark) + 1 # denotes the point which is not a keypoint

        if h == new_h and w == new_w:
            ret_d_img = d_img
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            ret_d_img = d_img[:, top: top + new_h, left: left + new_w]
            if develop:
                for idx in range(len(landmark)):
                    if top <= landmark[idx, 1] <= top+new_h and left <= landmark[idx, 0] <= left+new_w:
                        filtered_landmark.append(idx)
                    else:
                        not_landmark.append(landmark[idx,:])
                if len(filtered_landmark) < self.num_landmark:
                    res = self.num_landmark - len(filtered_landmark)
                    points = filtered_landmark
                    points += [other_point] * res
                else:
                    points_idx = np.random.choice(len(filtered_landmark), size=self.num_landmark, replace=False)
                    points += [filtered_landmark[i] for i in points_idx]

            if self.normalization:
                points = np.array([x for x in points])
                array_min, array_max = points.min(), points.max()
                points = ((points - array_min) / (array_max - array_min + 1e-6)) * 2 - 1


        sample = {
            'd_img_org': ret_d_img,
            'score': score,
            'landmark_org': points
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        landmark_org = sample['landmark_org']

        d_img = (d_img - self.mean) / self.var
        sample = {'d_img_org': d_img, 'score': score, 'landmark_org': landmark_org}


        return sample

class RandHorizontalFlip(object):
    def __init__(self, prob_aug):
        self.prob_aug = prob_aug

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        landmark_org = sample['landmark_org']

        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'score': score,
            'landmark_org':landmark_org
        }

        return sample


class RandRotation(object):
    def __init__(self, prob_aug=0.5):
        self.prob_aug = prob_aug
        self.aug_count = 0

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']

        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if prob_lr > 0.5:
            p = np.array([0.33, 0.33, 0.34])
            idx = np.random.choice([1, 2, 3], p=p.ravel())
            d_img = np.rot90(d_img, idx, axes=(1, 2)).copy()
            self.aug_count += 1
        
        sample = {
            'd_img_org': d_img,
            'score': score,
            'aug_count': self.aug_count
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        landmark_org = sample['landmark_org']

        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'score': score,
            'landmark_org': landmark_org
        }


        return sample
    

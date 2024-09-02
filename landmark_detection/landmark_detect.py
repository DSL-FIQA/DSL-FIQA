import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN
import json
from postprocess import alignment_procedure, rotate_facial_area
from tqdm import tqdm
import torch.nn.functional as F

def visualize_mask(image_tensor, mask):
    image = image_tensor.numpy().transpose(1, 2, 0) * 255 
    image = image.astype(np.uint8)
    
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mapping = np.array([
        [0, 0, 0],
        [0, 153, 255],
        [102, 255, 153],
        [0, 204, 153],
        [255, 255, 102],
        [255, 255, 204],
        [255, 153, 0],
        [255, 102, 255],
        [102, 0, 51],
        [255, 204, 255],
        [255, 0, 102]
    ])
    
    for index, color in enumerate(color_mapping):
        color_mask[mask == index] = color

    overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    return overlayed_image, image, color_mask

def visualize_landmarks(im, landmarks, color=255, thickness=5, eye_radius=3):
    im = im.permute(1, 2, 0).numpy()
    im = (im * 255).astype(np.uint8)
    im = np.ascontiguousarray(im)
    landmarks = landmarks.squeeze(0).numpy().astype(np.int32)
    for (x, y) in landmarks:
        cv2.circle(im, (x,y), eye_radius, color, thickness)
    return im

def visualize_head_pose(img, euler, tdx=None, tdy=None, size = 100):
    pitch, yaw, roll = euler[0], euler[1], euler[2]

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,255,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(255,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,255),2)
    return img

def denorm_points(points, h, w, align_corners=False):
    if align_corners:
        denorm_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(points).view(1, 1, 2)
    else:
        denorm_points = ((points + 1) * torch.tensor([w, h], dtype=torch.float32).to(points).view(1, 1, 2) - 1) / 2

    return denorm_points

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean 
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def adjust_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin_percentage):
    width = x_max - x_min
    height = y_max - y_min
    
    increase_width = width * (margin_percentage / 100.0) 
    increase_height = height * (margin_percentage / 100.0) 
    
    x_min_adjusted = max(0, x_min - increase_width) 
    y_min_adjusted = max(0, y_min - increase_height)
    x_max_adjusted = min(image_width, x_max + increase_width)
    y_max_adjusted = min(image_height, y_max + increase_height)
    
    return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted

def crop_and_align(args, model):
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(args.image_path)
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    
    transforms_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(args.img_size, args.img_size), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
  
    try:
        x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
        x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height, margin_percentage=args.bbox_margin_percentage)
        image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        image = transforms_image(image)
        task = torch.tensor([1])

        data = {'image': image, 'label': {"segmentation":torch.zeros([args.img_size, args.img_size]), "lnm_seg": torch.zeros([5, 2]),"landmark": torch.zeros([68, 2]), "headpose": torch.zeros([3]), "attribute": torch.zeros([40]), "a_g_e": torch.zeros([3]), 'visibility': torch.zeros([29])}, 'task': task}
        images, labels, tasks = data["image"], data["label"], data["task"]
        images = images.unsqueeze(0).to(device=device)
        
        for k in labels.keys():
            labels[k] = labels[k].unsqueeze(0).to(device=device)
            
        tasks = tasks.to(device=device)

        landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = model(images, labels, tasks)

        image = unnormalize(images[0].detach().cpu())
        denorm_landmarks = denorm_points(landmark_output.view(-1,68,2)[0], args.img_size, args.img_size)
        denorm_landmarks = denorm_landmarks.detach().cpu()
        # breakpoint()
        
        xmin, xmax = denorm_landmarks[0, :, 0].min(), denorm_landmarks[0, :, 0].max()
        ymin, ymax = denorm_landmarks[0, :, 1].min(), denorm_landmarks[0, :, 1].max()
        x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin
        
        im = image.permute(1, 2, 0).numpy()
        im = (im * 255).astype(np.uint8)
        im = np.ascontiguousarray(im)

        nose = denorm_landmarks[0, 33, :].tolist()
        left_eye = denorm_landmarks[0, 47, :].tolist()
        right_eye = denorm_landmarks[0, 38, :].tolist()
        
        aligned_img, rotate_angle, rotate_direction = alignment_procedure(
            img=im, left_eye=right_eye, right_eye=left_eye, nose=nose
        )
        
        # find new facial area coordinates after alignment
        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
            (0, 0, args.img_size, args.img_size), rotate_angle, rotate_direction, (args.img_size, args.img_size)
        )

        facial_img = aligned_img[
            int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
        ]    
        facial_img = cv2.resize(facial_img, (args.img_size, args.img_size))
        output_image = Image.fromarray(facial_img)

        return output_image

    except: 
        output_image = Image.open(args.image_path)
        return output_image

def landmark_detection(args, model, aligned_img):
    transforms_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    mtcnn = MTCNN(keep_all=True)
    image = aligned_img
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    
    args.landmark_results_path = os.path.join(args.results_path, 'landmark')
    args.aligned_results_path = os.path.join(args.results_path, 'img')
    os.makedirs(args.landmark_results_path, exist_ok=True)
    os.makedirs(args.aligned_results_path, exist_ok=True)

    try:
        image = transforms_image(image)
        task = torch.tensor([1])

        data = {'image': image, 'label': {"segmentation":torch.zeros([224, 224]), "lnm_seg": torch.zeros([5, 2]),"landmark": torch.zeros([68, 2]), "headpose": torch.zeros([3]), "attribute": torch.zeros([40]), "a_g_e": torch.zeros([3]), 'visibility': torch.zeros([29])}, 'task': task}
        images, labels, tasks = data["image"], data["label"], data["task"]
        images = images.unsqueeze(0).to(device=device)
        
        for k in labels.keys():
            labels[k] = labels[k].unsqueeze(0).to(device=device)
            
        tasks = tasks.to(device=device)

        landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = model(images, labels, tasks)

        image = unnormalize(images[0].detach().cpu())

        denorm_landmarks = denorm_points(landmark_output.view(-1,68,2)[0], 224, 224)
        denorm_landmarks = denorm_landmarks.detach().cpu()
                
        im = image.permute(1, 2, 0).numpy()
        im = (im * 255).astype(np.uint8)
        im = np.ascontiguousarray(im)   

        resized_image = F.interpolate(image.unsqueeze(0), size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
        final_image = resized_image.squeeze(0)
        
        scale_factor = args.img_size / 224
        denorm_landmarks = denorm_landmarks * scale_factor            
        
        final_save_path = os.path.join(args.landmark_results_path, args.image_path.split('/')[-1][:-4]+'.npy')
        output_landmark = denorm_landmarks[0].numpy()
        np.save(final_save_path, output_landmark)        
        image_save_path = os.path.join(args.aligned_results_path, args.image_path.split('/')[-1][:-4]+'.png')
        aligned_img.save(image_save_path)
                       
    except: 
        image_save_path = os.path.join(args.aligned_results_path, args.image_path.split('/')[-1][:-4]+'.png')
        aligned_img.save(image_save_path)
        print('No face detected in {} !!! '.format(args.image_path.split('/')[-1]))
        breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Provide absolute path to your weights file", default='ckpts/model.pt')
    parser.add_argument("--image_path", type=str, help="Provide absolute path to the image you want to perform inference on", default='../dataset/custom/unprocess')
    parser.add_argument("--results_path", type=str, help="Provide path to the folder where results need to be saved", default='../dataset/custom')
    parser.add_argument("--task", type=str, help="parsing" or "landmarks" or "headpose" or "attributes" or "age_gender_race" or "visibility", default="landmarks")
    parser.add_argument("--gpu_num", type=str, help="Provide the gpu number", default='0')
    parser.add_argument("--img_size", type=int, help="Resized input size", default=512)
    parser.add_argument("--bbox_margin_percentage", type=int, help="Margin percentage", default=80)
    args = parser.parse_args()  
    
    print(args)
    print('==> All results will be saved to {} <=='.format(args.results_path))
    image_list = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path)]
    image_list = sorted(image_list)

    device = "cuda:" + str(args.gpu_num)
    model = FaceXFormer().to(device)
    weights_path = args.model_path
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict_backbone'])

    model.eval()
        
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    for image_path in tqdm(image_list):
        args.image_path = image_path
        aligned_img = crop_and_align(args, model)
        landmark_detection(args, model, aligned_img)


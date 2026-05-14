import os
import json
import cv2
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from src.feature_extractor import compute_features

class TripletFSC147DatasetV2(Dataset):
    def __init__(self, json_path, split_json_path, image_dir, features_dir, split_name='train'):
        self.image_dir = image_dir
        self.features_dir = features_dir
        
        with open(split_json_path, 'r') as f:
            splits = json.load(f)
            
        allowed_images = splits.get(split_name, [])
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.valid_images = []
        for img_name in allowed_images:
            if img_name not in raw_data: continue
            data = raw_data[img_name]
            img_path = os.path.join(image_dir, img_name)
            feat_path = os.path.join(features_dir, img_name.replace('.jpg', '.npy'))
            if os.path.exists(img_path) and os.path.exists(feat_path):
                self.valid_images.append({'img_name': img_name, 'data': data})

    def __len__(self): return len(self.valid_images)

    def __getitem__(self, idx):
        item = self.valid_images[idx]
        img_name, data = item['img_name'], item['data']
        
        feat_path = os.path.join(self.features_dir, img_name.replace('.jpg', '.npy'))
        anchor_vector = random.choice(np.load(feat_path)) 
        
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img_h, img_w, _ = img.shape
        
        boxes = data['box_examples_coordinates']
        avg_w = int(np.mean([max([p[0] for p in b]) - min([p[0] for p in b]) for b in boxes]))
        avg_h = int(np.mean([max([p[1] for p in b]) - min([p[1] for p in b]) for b in boxes]))
        
        # Positive
        points = data['points']
        px, py = int(random.choice(points)[0]), int(random.choice(points)[1])
        p_patch = img[max(0, py - avg_h//2):min(img_h, py + avg_h//2), max(0, px - avg_w//2):min(img_w, px + avg_w//2)]
        positive_vector = compute_features(p_patch)
        
        # Hard Negative
        best_neg_patch, highest_var = None, -1
        for _ in range(15):
            nx, ny = random.randint(avg_w//2, img_w - avg_w//2), random.randint(avg_h//2, img_h - avg_h//2)
            if min([math.hypot(nx - pt[0], ny - pt[1]) for pt in points]) > max(avg_w, avg_h):
                t_patch = img[max(0, ny - avg_h//2):min(img_h, ny + avg_h//2), max(0, nx - avg_w//2):min(img_w, nx + avg_w//2)]
                if t_patch.shape[0] > 0 and t_patch.shape[1] > 0:
                    var = cv2.Laplacian(cv2.cvtColor(t_patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if var > highest_var:
                        highest_var, best_neg_patch = var, t_patch

        if best_neg_patch is None: best_neg_patch = img[0:avg_h, 0:avg_w]
        negative_vector = compute_features(best_neg_patch)
        
        return torch.tensor(anchor_vector, dtype=torch.float32), \
               torch.tensor(positive_vector, dtype=torch.float32), \
               torch.tensor(negative_vector, dtype=torch.float32)
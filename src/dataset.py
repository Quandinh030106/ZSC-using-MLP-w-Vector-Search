import os
import sys
import json
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FSC147FeatureStore:
    def __init__(self, json_path, split_json_path, image_dir, features_dir, split_name='train'):
        self.image_dir = image_dir
        self.features_dir = features_dir

        with open(split_json_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.items = []
        for img_name in splits.get(split_name, []):
            feat_path = os.path.join(features_dir, img_name.replace('.jpg', '.npy'))
            img_path = os.path.join(image_dir, img_name)
            if img_name in raw_data and os.path.exists(feat_path) and os.path.exists(img_path):
                self.items.append({
                    'img_name': img_name,
                    'img_path': img_path,
                    'feat_path': feat_path,
                    'data': raw_data[img_name],
                })

    def __len__(self):
        return len(self.items)

    def load_features(self):
        features = []
        for item in self.items:
            arr = np.load(item['feat_path']).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            features.append(arr)

        if not features:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(features)

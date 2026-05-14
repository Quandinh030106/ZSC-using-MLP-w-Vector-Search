import cv2
import json
import numpy as np
import os
from tqdm import tqdm
from src.config import CFG

HOG_DESCRIPTOR = cv2.HOGDescriptor(
    _winSize=CFG.PATCH_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def compute_features(patch):
    """Hàm trích xuất HOG+HSV 1812 chiều"""
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros(CFG.FEATURE_DIM, dtype=np.float32)
    
    patch_resized = cv2.resize(patch, CFG.PATCH_SIZE)
    
    gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
    hog_vector = HOG_DESCRIPTOR.compute(gray).flatten()
    
    hsv = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2HSV)
    hist_H = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_S = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_V = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    cv2.normalize(hist_H, hist_H)
    cv2.normalize(hist_S, hist_S)
    cv2.normalize(hist_V, hist_V)
    
    color_vector = np.concatenate((hist_H, hist_S, hist_V))
    return np.concatenate((hog_vector, color_vector)).astype(np.float32)

if __name__ == "__main__":
    os.makedirs(CFG.FEATURES_DIR, exist_ok=True)
    with open(CFG.JSON_PATH, 'r') as f:
        annotations = json.load(f)

    print("Bắt đầu trích xuất Đặc trưng V2...")
    for img_name, data in tqdm(annotations.items()):
        img_path = os.path.join(CFG.IMAGE_DIR, img_name)
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        boxes = data.get('box_examples_coordinates', [])
        image_features = []
        
        for box in boxes:
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))
            xmin, ymin = max(0, xmin), max(0, ymin)
            
            patch = img[ymin:ymax, xmin:xmax]
            vector = compute_features(patch)
            image_features.append(vector)
        
        if len(image_features) > 0:
            np.save(os.path.join(CFG.FEATURES_DIR, img_name.replace('.jpg', '.npy')), np.array(image_features))
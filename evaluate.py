import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

from src.config import CFG
from src.model import SiameseResidualMLP
from src.vector_search import FAISSSearchEngine
from src.feature_extractor import compute_features
from src.utils import non_max_suppression, sliding_window

def evaluate_split(split_name, model, device, annotations, splits):
    
    image_names = splits.get(split_name, [])
    if not image_names:
        print(f"Không tìm thấy ảnh nào trong tập {split_name}.")
        return None, None

    print(f"\n ĐANG ĐÁNH GIÁ TẬP: {split_name.upper()} ({len(image_names)} ảnh)")
    
    absolute_errors = []
    squared_errors = []

    # Quét qua từng bức ảnh trong tập Test
    for img_name in tqdm(image_names, desc=f"Evaluating {split_name}"):
        if img_name not in annotations:
            continue
            
        data = annotations[img_name]
        img_path = os.path.join(CFG.IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Số lượng đếm Ground Truth
        ground_truth_count = len(data.get('points', []))
        
        # Lấy tọa độ các vật mẫu (FSC-147 thường cho 3 ô mẫu)
        boxes = data.get('box_examples_coordinates', [])
        if len(boxes) == 0:
            continue
            
        query_feats = []
        win_widths, win_heights = [], []
        
        for box in boxes:
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            xmin, xmax = max(0, int(min(x_coords))), int(max(x_coords))
            ymin, ymax = max(0, int(min(y_coords))), int(max(y_coords))
            
            if xmax <= xmin or ymax <= ymin: continue
                
            patch = img[ymin:ymax, xmin:xmax]
            query_feats.append(compute_features(patch))
            win_widths.append(xmax - xmin)
            win_heights.append(ymax - ymin)
            
        if not query_feats:
            continue
            
        # Lấy trung bình kích thước cửa sổ trượt
        win_w = int(np.mean(win_widths))
        win_h = int(np.mean(win_heights))
        step_size = max(4, min(win_w, win_h) // 4) 
        
        # Vector Query trung bình
        avg_query_feat = np.mean(query_feats, axis=0)
        query_tensor = torch.tensor(avg_query_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            query_vector = model(query_tensor).cpu().numpy()

        # Quét cửa sổ trượt
        # Quét ảnh ở 3 kích thước: Nhỏ (80%), Vừa (100%), To (120%)
        scales = [0.8, 1.0, 1.2]
        
        all_patches = []
        all_patch_boxes = []
        
        for scale in scales:
            scaled_w = int(win_w * scale)
            scaled_h = int(win_h * scale)
            
            if scaled_w < 10 or scaled_h < 10 or scaled_w > img.shape[1] or scaled_h > img.shape[0]:
                continue
                
            step_size = max(4, min(scaled_w, scaled_h) // 4) 
            patches, patch_boxes = sliding_window(img, (scaled_w, scaled_h), step_size)
            
            all_patches.extend(patches)
            all_patch_boxes.extend(patch_boxes)
            
        if not all_patches:
            continue
            
        # Trích xuất và Nén
        db_feats = [compute_features(p) for p in patches]
        db_tensor = torch.tensor(np.array(db_feats)).to(device)
        
        db_vectors = []
        with torch.no_grad():
            for i in range(0, len(db_tensor), 512):
                db_vectors.append(model(db_tensor[i:i+512]).cpu().numpy())
        db_vectors = np.vstack(db_vectors)

        # FAISS Search
        searcher = FAISSSearchEngine(dimension=CFG.LATENT_DIM, metric='cosine')
        searcher.build_database(db_vectors)
        scores, indices = searcher.search_exemplar(query_vector, top_k=len(db_vectors))

        # Lọc Ngưỡng & NMS
        filtered_boxes = [patch_boxes[idx] for score, idx in zip(scores, indices) if score > CFG.SIMILARITY_THRESHOLD]
        filtered_scores = [score for score in scores if score > CFG.SIMILARITY_THRESHOLD]

        keep = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=CFG.NMS_THRESHOLD)
        
        # SỐ ĐẾM DỰ ĐOÁN
        pred_count = len(keep)
        
        # Tính toán sai số
        absolute_errors.append(abs(pred_count - ground_truth_count))

        squared_errors.append((pred_count - ground_truth_count) ** 2)

    # Tính MAE và RMSE
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    return mae, rmse

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SiameseResidualMLP(input_dim=CFG.FEATURE_DIM, latent_dim=CFG.LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Load file Annotations & Splits
    with open(CFG.JSON_PATH, 'r') as f:
        annotations = json.load(f)
    with open(CFG.SPLIT_PATH, 'r') as f:
        splits = json.load(f)

    # Chạy đánh giá
    test_mae, test_rmse = evaluate_split('test', model, device, annotations, splits)
    coco_mae, coco_rmse = evaluate_split('test_coco', model, device, annotations, splits)

    # IN KẾT QUẢ BÁO CÁO
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (FINAL REPORT)")
    print("="*50)
    if test_mae is not None:
        print(f"Tập [TEST]:")
        print(f"   - MAE  : {test_mae:.2f}")
        print(f"   - RMSE : {test_rmse:.2f}")
    
    print("-" * 50)
    if coco_mae is not None:
        print(f"Tập [TEST_COCO]:")
        print(f"   - MAE  : {coco_mae:.2f}")
        print(f"   - RMSE : {coco_rmse:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
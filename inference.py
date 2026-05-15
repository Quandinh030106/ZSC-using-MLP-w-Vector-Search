import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import CFG
from src.model import SiameseResidualMLP
from src.vector_search import FAISSSearchEngine
from src.feature_extractor import compute_features
from src.utils import non_max_suppression
from src.utils import sliding_window

IMAGE_PATH = 'data/images_384_VarV2/685.jpg' 
EXEMPLAR_BOX = [345, 219, 391, 269]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SiameseResidualMLP(input_dim=CFG.FEATURE_DIM, latent_dim=CFG.LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # ĐỌC ẢNH VÀ NÉN QUERY
    img = cv2.imread(IMAGE_PATH)
    img_draw = img.copy()
    xmin, ymin, xmax, ymax = EXEMPLAR_BOX
    
    query_feat = compute_features(img[ymin:ymax, xmin:xmax])
    with torch.no_grad():
        query_vector = model(torch.tensor(query_feat).unsqueeze(0).to(device)).cpu().numpy()

    # MULTI-SCALE SLIDING WINDOW
    win_w, win_h = xmax - xmin, ymax - ymin
    scales = [0.8, 1.0, 1.2]
    
    all_patches = []
    all_boxes = []
    
    for scale in scales:
        scaled_w = int(win_w * scale)
        scaled_h = int(win_h * scale)
        
        # Bỏ qua nếu cửa sổ quá nhỏ hoặc tràn viền ảnh
        if scaled_w < 10 or scaled_h < 10 or scaled_w > img.shape[1] or scaled_h > img.shape[0]:
            continue
            
        step_size = max(4, min(scaled_w, scaled_h) // 4) 
        patches, boxes = sliding_window(img, (scaled_w, scaled_h), step_size)
        all_patches.extend(patches)
        all_boxes.extend(boxes)
    
    print(f"   -> Đã cắt ra tổng cộng {len(all_patches)} ô ảnh.")

    # TRÍCH XUẤT VÀ NÉN VECTOR
    db_feats = [compute_features(p) for p in tqdm(all_patches, desc="[1/3] Trích xuất HOG+HSV")]
    db_tensor = torch.tensor(np.array(db_feats)).to(device)
    
    db_vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(db_tensor), 512), desc="[2/3] Mạng MLP Nén Vector"):
            db_vectors.append(model(db_tensor[i:i+512]).cpu().numpy())
    db_vectors = np.vstack(db_vectors)

    # TÌM KIẾM BẰNG FAISS(VECTOR SEARCH)
    searcher = FAISSSearchEngine(dimension=CFG.LATENT_DIM, metric='cosine')
    searcher.build_database(db_vectors)
    scores, indices = searcher.search_exemplar(query_vector, top_k=len(db_vectors))

    # LỌC NGƯỠNG & NMS
    filtered_boxes = [all_boxes[idx] for score, idx in zip(scores, indices) if score > CFG.SIMILARITY_THRESHOLD]
    filtered_scores = [score for score in scores if score > CFG.SIMILARITY_THRESHOLD]

    print("[3/3] Chạy NMS lọc nhiễu...")
    keep = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=CFG.NMS_THRESHOLD)
    
    # 7. KẾT QUẢ VÀ VẼ HÌNH
    COUNT = len(keep)
    print(f"\nTỔNG SỐ ĐẾM DỰ ĐOÁN: {COUNT}\n")

    for i in keep:
        x1, y1, x2, y2 = filtered_boxes[i]
        # Vẽ khung tìm được bằng màu đỏ
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Vẽ lại khung mẫu bằng màu xanh lá
    cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.title(f"Zero-Shot Counting (Multi-Scale) | Predicted: {COUNT}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
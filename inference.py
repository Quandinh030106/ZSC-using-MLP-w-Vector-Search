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


IMAGE_PATH = 'data/images_384_VarV2/685.jpg' 
EXEMPLAR_BOX = [345, 219, 391, 269]

def sliding_window(image, window_size, step_size):
    h, w = image.shape[:2]
    win_w, win_h = window_size
    patches, boxes = [], []
    for y in range(0, h - win_h + 1, step_size):
        for x in range(0, w - win_w + 1, step_size):
            patches.append(image[y:y+win_h, x:x+win_w])
            boxes.append([x, y, x + win_w, y + win_h])
    return patches, boxes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. LOAD MODEL
    model = SiameseResidualMLP(input_dim=CFG.FEATURE_DIM, latent_dim=CFG.LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CFG.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 2. ĐỌC ẢNH VÀ NÉN QUERY
    img = cv2.imread(IMAGE_PATH)
    img_draw = img.copy()
    xmin, ymin, xmax, ymax = EXEMPLAR_BOX
    
    query_feat = compute_features(img[ymin:ymax, xmin:xmax])
    with torch.no_grad():
        query_vector = model(torch.tensor(query_feat).unsqueeze(0).to(device)).cpu().numpy()

    # 3. SLIDING WINDOW & EXTRACT BATCH
    win_w, win_h = xmax - xmin, ymax - ymin
    step_size = max(4, min(win_w, win_h) // 4) 
    patches, boxes = sliding_window(img, (win_w, win_h), step_size)
    
    db_feats = [compute_features(p) for p in tqdm(patches, desc="[1/3] Trích xuất HOG+HSV")]
    db_tensor = torch.tensor(np.array(db_feats)).to(device)
    
    db_vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(db_tensor), 512), desc="[2/3] Mạng MLP Nén Vector"):
            db_vectors.append(model(db_tensor[i:i+512]).cpu().numpy())
    db_vectors = np.vstack(db_vectors)

    # 4. TÌM KIẾM BẰNG fAISS
    searcher = FAISSSearchEngine(dimension=CFG.LATENT_DIM, metric='cosine')
    searcher.build_database(db_vectors)
    scores, indices = searcher.search_exemplar(query_vector, top_k=len(db_vectors))

    # 5. LỌC NGƯỠNG & NMS
    filtered_boxes = [boxes[idx] for score, idx in zip(scores, indices) if score > CFG.SIMILARITY_THRESHOLD]
    filtered_scores = [score for score in scores if score > CFG.SIMILARITY_THRESHOLD]

    print("[3/3] Chạy NMS lọc nhiễu...")
    keep = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=CFG.NMS_THRESHOLD)
    
    # 6. KẾT QUẢ
    COUNT = len(keep)
    print(f"\nTỔNG SỐ ĐẾM: {COUNT}\n")

    for i in keep:
        x1, y1, x2, y2 = filtered_boxes[i]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.title(f"Zero-Shot Predicted: {COUNT}")
    plt.show()

if __name__ == "__main__":
    main()
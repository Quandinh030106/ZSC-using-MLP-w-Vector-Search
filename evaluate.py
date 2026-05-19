import os
import cv2
import json
import numpy as np
from tqdm import tqdm

from src.config import CFG
from src.counting import load_count_calibration, predict_count_from_annotation
from src.model import PCAWhiteningModel
from src.utils import configure_utf8_output

def evaluate_split(split_name, model, annotations, splits, calibration=None):
    
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

        ground_truth_count = len(data.get('points', []))
        pred_count, _ = predict_count_from_annotation(img, model, data, calibration=calibration)
        
        # Tính toán sai số
        absolute_errors.append(abs(pred_count - ground_truth_count))

        squared_errors.append((pred_count - ground_truth_count) ** 2)

    # Tính MAE và RMSE
    if not absolute_errors:
        return None, None

    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    return mae, rmse

def main():
    configure_utf8_output()
    model_path = CFG.resolve_best_model_path()
    model = PCAWhiteningModel.load(model_path)
    print(f"Loaded PCA/Whitening model: {model_path}")

    calibration_path = CFG.resolve_count_calibration_path()
    calibration = load_count_calibration(calibration_path)
    if calibration is None:
        print("No count calibration found. Using default threshold/NMS count.")
    else:
        print(f"Loaded count calibration: {calibration_path}")
        print(
            "Calibration params: "
            f"threshold={calibration['similarity_threshold']:.2f}, "
            f"nms={calibration['nms_threshold']:.2f}, "
            f"scale={calibration['scale']:.4f}, "
            f"bias={calibration['bias']:.4f}, "
            f"val_mae={calibration['val_mae']:.2f}"
        )

    # Load file Annotations & Splits
    with open(CFG.JSON_PATH, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    with open(CFG.SPLIT_PATH, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    # Chạy đánh giá
    test_mae, test_rmse = evaluate_split('test', model, annotations, splits, calibration=calibration)
    coco_mae, coco_rmse = evaluate_split('test_coco', model, annotations, splits, calibration=calibration)

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

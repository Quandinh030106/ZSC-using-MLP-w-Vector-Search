import os
import cv2
import matplotlib.pyplot as plt

from src.config import CFG
from src.counting import (
    count_from_candidates,
    load_count_calibration,
    build_detection_candidates,
)
from src.model import PCAWhiteningModel
from src.utils import configure_utf8_output
from src.utils import non_max_suppression

IMAGE_PATH = os.path.join(CFG.IMAGE_DIR, '685.jpg')
EXEMPLAR_BOX = [345, 219, 391, 269]


def main():
    configure_utf8_output()
    model_path = CFG.resolve_best_model_path()
    model = PCAWhiteningModel.load(model_path)
    print(f"Loaded PCA/Whitening model: {model_path}")

    calibration = load_count_calibration(CFG.resolve_count_calibration_path())
    if calibration is not None:
        print(
            "Loaded count calibration: "
            f"threshold={calibration['similarity_threshold']:.2f}, "
            f"nms={calibration['nms_threshold']:.2f}, "
            f"scale={calibration['scale']:.4f}, "
            f"bias={calibration['bias']:.4f}"
        )

    # ĐỌC ẢNH VÀ NÉN QUERY
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy hoặc không đọc được ảnh: {IMAGE_PATH}")

    img_draw = img.copy()
    xmin, ymin, xmax, ymax = EXEMPLAR_BOX
    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin or xmax > img.shape[1] or ymax > img.shape[0]:
        raise ValueError(f"EXEMPLAR_BOX không hợp lệ với kích thước ảnh: {EXEMPLAR_BOX}")

    scores, all_boxes = build_detection_candidates(img, model, [EXEMPLAR_BOX])
    print(f"   -> Đã tạo {len(scores)} candidate windows.")
    if len(scores) == 0:
        raise ValueError("Không tạo được candidate window nào từ EXEMPLAR_BOX hiện tại.")

    threshold = calibration["similarity_threshold"] if calibration is not None else CFG.SIMILARITY_THRESHOLD
    nms_threshold = calibration["nms_threshold"] if calibration is not None else CFG.NMS_THRESHOLD
    raw_count = count_from_candidates(scores, all_boxes, threshold, nms_threshold)

    print("[3/3] Chạy NMS lọc nhiễu...")
    candidate_pairs = [(float(score), idx) for idx, score in enumerate(scores) if float(score) > threshold]
    if not candidate_pairs:
        fallback_k = min(50, len(scores))
        candidate_pairs = [(float(scores[idx]), idx) for idx in range(fallback_k)]
    filtered_scores = [score for score, _ in candidate_pairs]
    filtered_boxes = [all_boxes[idx] for _, idx in candidate_pairs]
    keep = non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=nms_threshold)
    
    # 7. KẾT QUẢ VÀ VẼ HÌNH
    if calibration is None:
        COUNT = raw_count
    else:
        COUNT = int(max(0, round(calibration["scale"] * float(raw_count) + calibration["bias"])))
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

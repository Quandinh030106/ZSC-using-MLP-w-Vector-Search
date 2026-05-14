import numpy as np

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Thuật toán Non-Maximum Suppression (NMS)
    - boxes: Danh sách các khung hình chữ nhật [xmin, ymin, xmax, ymax]
    - scores: Cosine Similarity của từng bbox
    - iou_threshold: Ngưỡng chồng lấp
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Lấy tọa độ
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Tính diện tích của tất cả các hộp
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sắp xếp các hộp
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0] # Lấy hộp có điểm cao nhất
        keep.append(i)

        # Tính toán tọa độ phần giao nhau của hộp i với các hộp còn lại
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Tính diện tích phần giao nhau
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Tính IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Giữ lại IoU < Threshhold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
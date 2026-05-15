#  Zero-Shot Object Counting (ZSC) using Classical CV & Siamese MLP

Đồ án môn học Machine Learning - Giải quyết bài toán Đếm vật thể Zero-shot (ZSC) **không sử dụng mạng Tích chập (CNN) hay Transformer (ViT)**. 

Dự án bao gồm **Feature Extracting của Classical CV**, **Neural Network kết hơp với Siamese Residual MLP** và **Vector Search bằng FAISS**

##  Các đặc điểm của dự án
### 1. Multi-modal Feature Extraction (Trích xuất đặc trưng đa phương thức)
Thay vì dùng điểm ảnh thô (pixel), hệ thống trích xuất vector 1812 chiều từ mỗi ô ảnh (được chuẩn hóa về 64x64):
*   **HOG (Histogram of Oriented Gradients) - 1764 chiều:** Nắm bắt hình dáng học và đường viền (Shape & Edges) của vật thể trên ảnh xám.
*   **HSV Color Histogram - 48 chiều:** Chuyển đổi sang không gian màu HSV để bóc tách yếu tố ánh sáng, chia làm 16 bins cho mỗi kênh.

### 2. Multi-scale Sliding Window (Cửa sổ trượt đa kích thước)
*   **Vấn đề:** Các vật thể trong cùng một bức ảnh có kích thước to nhỏ khác nhau do hiệu ứng xa gần. Cửa sổ trượt cố định sẽ cắt lẹm hoặc chứa quá nhiều nền.
*   **Giải pháp:** Tự động lấy tỷ lệ từ khung vật thể mẫu (Exemplar Box), sau đó quét qua ảnh bằng **nhiều tỷ lệ phóng to/thu nhỏ khác nhau** (0.8x, 1.0x, 1.2x). Điều này giúp model thích ứng và không bỏ sót vật thể xa-gần.

### 3. Deep Siamese Residual MLP (~11 Triệu tham số)
*   Mạng nơ-ron thuần `Dense Layer` nhưng được thiết kế học sâu (Deep Learning) với kiến trúc Phễu: 1812 → 1536 → 768 → 256 → 128.
*   **Skip Connection (Residual):** Bổ sung kết nối tắt $F(x) + x$ để tránh hiện tượng triệt tiêu đạo hàm khi mạng sâu.
*   **LayerNorm & High Dropout:** Sử dụng `LayerNorm` (tối ưu cho vector 1D hơn BatchNorm) và `Dropout(0.5)` kết hợp `Weight Decay` để chống Overfitting.

### 4. Hard Negative Mining with Laplacian Variance
*   Hàm **Triplet Margin Loss** cần các bộ ba (Anchor, Positive, Negative) để học.
*   Thay vì lấy vùng nền (Negative) ngẫu nhiên (dễ lấy trúng bầu trời trơn láng), hệ thống dùng thuật toán **Variance of Laplacian** để đo độ phức tạp gờ mép của hàng loạt vùng nền. Khu vực nào càng rắc rối, hệ thống càng ưu tiên bốc làm Negative để mô hình học cách phân biệt sự khác nhau.

### 5. Vector Search bằng FAISS & NMS
*   Lớp cuối cùng của MLP áp dụng **L2 Normalization**, ép mọi vector lên bề mặt hình cầu đơn vị.
*   Đưa hàng chục ngàn vector của các ô cửa sổ trượt vào **FAISS**. Việc tính `Inner Product` trên FAISS tương đương với việc đo **Cosine Similarity**.
*   Cuối cùng, áp dụng thuật toán **Non-Maximum Suppression (NMS)** dựa trên ngưỡng IoU để loại bỏ các hộp giới hạn (Bounding Boxes) chồng chéo.

## Cấu trúc thư mục
```text
 ZSC-Classical-MLP
 ┣  src
 ┃ ┣  config.py
 ┃ ┣  feature_extractor.py
 ┃ ┣  dataset.py
 ┃ ┣  model.py
 ┃ ┣  train.py
 ┃ ┣  utils.py
 ┃ ┗  vector_search.py
 ┣  inference.py
 ┣  evaluate.py
 ┣  get_coords.py
 ┣  requirements.txt          
 ┗  README.md
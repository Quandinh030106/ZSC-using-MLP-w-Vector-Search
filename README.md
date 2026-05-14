# 🦅 Zero-Shot Object Counting (ZSC) using Classical CV & Siamese MLP

Đồ án môn học Machine Learning - Giải quyết bài toán Đếm vật thể Zero-shot (ZSC) mà **không sử dụng bất kỳ mạng Tích chập (CNN) hay Transformer (ViT) nào**. 

Dự án là sự kết hợp độc đáo giữa **Kỹ thuật trích xuất đặc trưng Cổ điển (Classical CV)**, **Mạng song sinh thuần Dense Layer (Siamese Residual MLP 11 triệu tham số)** và **Công cụ tìm kiếm Vector (FAISS)**.

## 🌟 Điểm nổi bật của Kiến trúc
1. **Feature Extraction:** Sử dụng HOG (Histogram of Oriented Gradients) kết hợp HSV Color Histogram để tạo ra vector đặc trưng 1812 chiều (Bắt trọn cả hình dáng và màu sắc).
2. **Siamese Residual MLP:** Mạng học sâu thuần Dense Layer áp dụng kỹ thuật *Skip Connection (Residual)* và *LayerNorm*, chống mất mát thông tin.
3. **Hard Negative Mining:** Thuật toán tính phương sai *Laplacian* tự động tìm các vùng nền phức tạp (mái ngói, cành cây...) để ép mô hình học phân biệt sắc nét.
4. **FAISS Vector Search:** Nén toàn bộ cửa sổ trượt thành Latent Space (128 chiều) và dùng FAISS (Meta) đo khoảng cách Cosine Similarity siêu tốc.

## 📂 Cấu trúc thư mục
```text
📦 ZSC-Classical-MLP
 ┣ 📂 src
 ┃ ┣ 📜 config.py
 ┃ ┣ 📜 feature_extractor.py
 ┃ ┣ 📜 dataset.py
 ┃ ┣ 📜 model.py
 ┃ ┣ 📜 train.py
 ┃ ┣ 📜 utils.py
 ┃ ┗ 📜 vector_search.py
 ┣ 📜 inference.py
 ┣ 📜 get_coords.py
 ┣ 📜 requirements.txt          
 ┗ 📜 README.md
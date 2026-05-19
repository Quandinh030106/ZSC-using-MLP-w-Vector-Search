# Zero-Shot Object Counting (ZSC) using Classical CV, PCA Whitening & FAISS

Đồ án Machine Learning giải quyết bài toán đếm vật thể zero-shot **không sử dụng CNN, Transformer, DNN hay neural network**.

Dự án dùng đặc trưng thị giác cổ điển, giảm chiều bằng PCA/Whitening, tìm kiếm vector bằng FAISS và lọc hộp bằng NMS.

## Pipeline

```text
HOG/HSV/LBP features
→ PCA/Whitening
→ L2 normalization
→ FAISS cosine search
→ NMS
→ Count
```

## Đặc Trưng

Mỗi patch ảnh được resize về `64x64` và trích xuất vector `2068` chiều:

* **HOG - 1764 chiều:** mô tả shape, edge và hướng gradient.
* **HSV histogram - 48 chiều:** 16 bins cho mỗi kênh H/S/V.
* **LBP - 256 chiều:** mô tả texture cục bộ bằng Local Binary Pattern.

## PCA/Whitening

`src/train.py` không huấn luyện neural network. File này chỉ fit PCA/Whitening trên feature exemplar của tập train, sau đó lưu:

```text
weights/pca_whitening_best.npz
weights/pca_whitening_best_checkpoint.npz
```

PCA giúp giảm nhiễu và nén feature. Whitening giúp các chiều chính có thang đo cân bằng hơn trước khi dùng cosine similarity.

## Inference & Evaluation

* `inference.py`: dùng một ảnh và một `EXEMPLAR_BOX` để đếm thử.
* `evaluate.py`: tự lấy exemplar boxes từ annotation FSC-147 và báo cáo MAE/RMSE trên `test` và `test_coco`.

## Thứ Tự Chạy

```powershell
pip install -r requirements.txt
python src/feature_extractor.py
python src/train.py
python inference.py
python evaluate.py
```

Trên Kaggle, code tự tìm dataset trong `/kaggle/input` và lưu output sinh ra vào `/kaggle/working`.

## Chạy Trên Kaggle

Sau khi upload dataset FSC147 và zip code này vào Kaggle, chạy trong một notebook cell:

```bash
CODE_DIR=$(find /kaggle/input -type f -name kaggle_install_and_run.sh -printf '%h\n' | head -n 1)
bash "$CODE_DIR/kaggle_install_and_run.sh"
```

Script mặc định dùng mode `auto`:

* Nếu chưa có model PCA, script tự chạy `feature_extractor.py`, rồi `train.py`.
* Nếu chưa có `count_calibration.npz`, script chạy `calibrate.py` trên validation để chọn threshold/NMS/count scale tốt hơn.
* Cuối cùng script chạy `evaluate.py`.

Nếu chỉ muốn evaluate và bắt lỗi ngay khi thiếu weight:

```bash
CODE_DIR=$(find /kaggle/input -type f -name kaggle_run.py -printf '%h\n' | head -n 1)
cd "$CODE_DIR"
python -B kaggle_run.py --mode evaluate
```

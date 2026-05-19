# README Chat Tu Luc Doi Sang PCA/Whitening

File nay ghi lai phan trao doi tu luc quyet dinh bo DNN/neural network va chuyen sang pipeline classical-only bang PCA/Whitening + FAISS.

---

## 1. Ly Do Doi Sang PCA

**Nguoi dung noi ro:** Thay/co cam luon DNN/neural network.

**Ket luan cua Codex:** Neu cam nghiem ngat neural network, khong nen dung:

- SiameseResidualMLP
- Pairwise MLP re-ranker
- Torch model
- Triplet/BCE training
- `.pth` weights

Huong classical hop ly nhat duoc chon:

```text
HOG + HSV + LBP
-> PCA/Whitening
-> L2 normalize
-> FAISS cosine search
-> NMS
-> Count
```

---

## 2. Noi Dung Trao Doi Chinh

### Nguoi dung

Neu thay/co cam luon DNN/neural network, thi lua chon on nhat tiep theo la PCA/Whitening + cosine similarity + FAISS, co the them LBP/Gabor de feature manh hon. OKE thi that su thay toi CAM LUON DNN/Neural Networks roi. Va ngoai PCA/Whitening + cosine similarity + FAISS thi con cai gi thuc su tot hon khong? Neu KHONG thi AP DUNG LUON CHO TOI voi PHUONG PHAP NAY DI. CO LEN.

### Codex

Tra loi rang trong rang buoc khong neural network, PCA/Whitening + cosine similarity + FAISS la lua chon on nhat va de bao ve nhat. Cac lua chon nhu SVM/Random Forest/LMNN/NCA co the can nhan, nhung voi bai toan sliding-window retrieval/counting tren FSC147, pipeline PCA/Whitening + FAISS don gian, classical, de giai thich va dong bo nhat.

---

## 3. Thay Doi Code Sau Khi Doi Sang PCA

### `src/model.py`

Thay model DNN bang:

```text
PCAWhiteningModel
```

Chuc nang:

- `fit(features)`
- `transform(features)`
- `save(path)`
- `load(path)`
- `truncated(n_components)`
- `_l2_normalize(vectors)`

Model luu bang `.npz`, khong dung `.pth`.

### `src/feature_extractor.py`

Feature moi:

```text
HOG: 1764 dim
HSV histogram: 48 dim
LBP: 256 dim
Total: 2068 dim
```

### `src/train.py`

Khong train neural network. Chi:

1. Load feature train/val.
2. Fit PCA/Whitening tren train features.
3. Thu cac PCA dim:

```text
64, 128, 256, 512
```

4. Chon best theo validation proxy loss.
5. Luu:

```text
weights/pca_whitening_best.npz
weights/pca_whitening_best_checkpoint.npz
```

### `evaluate.py`

Dung PCA model de transform query/candidate features, sau do:

```text
FAISS cosine search
-> threshold
-> NMS
-> count
-> MAE/RMSE
```

### `inference.py`

Dung mot anh va `EXEMPLAR_BOX` de dem thu bang pipeline PCA classical.

### `requirements.txt`

Khong con `torch`.

---

## 4. Thu Tu Chay Sau Khi Doi Sang PCA

Local:

```powershell
pip install -r requirements.txt
python src/feature_extractor.py
python src/train.py
python evaluate.py
```

Neu can chay mot anh rieng:

```powershell
python get_coords.py
python inference.py
```

---

## 5. Ket Qua Train PCA

Ket qua nguoi dung chay local:

```text
PCA dim 64:  val_loss=0.2475, acc=0.2446, var=0.6093
PCA dim 128: val_loss=0.2183, acc=0.2557, var=0.7307
PCA dim 256: val_loss=0.2000, acc=0.2506, var=0.8425
PCA dim 512: val_loss=0.1837, acc=0.2258, var=0.9288

Best PCA dim: 512
Best val loss: 0.1837
Best val accuracy: 0.2258
```

Codex giai thich: `val accuracy` o day chi la proxy de chon PCA dim, khong phai accuracy counting cuoi. Chi so counting chinh la MAE/RMSE.

---

## 6. Kaggle Runner Sau Khi Doi Sang PCA

Codex them:

```text
kaggle_run.py
kaggle_install_and_run.sh
```

Chay tren Kaggle bang Python cell:

```python
from pathlib import Path
import subprocess

matches = list(Path("/kaggle/input").rglob("kaggle_install_and_run.sh"))
assert matches, "Khong tim thay kaggle_install_and_run.sh"

script_path = matches[0]
print("Script:", script_path)

subprocess.run(["bash", str(script_path)], check=True)
```

Runner auto:

```text
Neu chua co PCA model:
  src/feature_extractor.py
  src/train.py

Neu chua co count calibration:
  calibrate.py

Sau do:
  evaluate.py
```

---

## 7. Ket Qua Evaluate Dau Tien Tren Kaggle

Nguoi dung chay evaluate va co ket qua:

```text
TEST:
  MAE  : 50.01
  RMSE : 83.10

TEST_COCO:
  MAE  : 54.77
  RMSE : 98.59
```

Danh gia cua Codex:

- MAE nay kha thap neu so voi model counting manh.
- Voi rang buoc no-neural-network, pipeline classical co gioi han.
- Van co the cai thien bang calibration co dien.

---

## 8. Them Calibration Co Dien

Codex them calibration khong dung neural network:

```text
validation set
-> thu nhieu similarity_threshold
-> thu nhieu NMS threshold
-> fit count = scale * raw_count + bias
-> luu count_calibration.npz
```

File moi:

```text
src/counting.py
calibrate.py
```

File sua:

```text
evaluate.py
inference.py
kaggle_run.py
src/config.py
```

---

## 9. Similarity Threshold La Gi?

`similarity_threshold` la nguong tren cosine similarity giua:

```text
query vector tu exemplar patch
candidate vector tu sliding-window patch
```

Sau PCA/Whitening:

```text
score = cosine_similarity(query_vector, candidate_vector)
```

Neu:

```text
score > similarity_threshold
```

thi candidate duoc giu lai de dua qua NMS.

Neu score thap hon threshold thi candidate bi loai.

`NMS threshold` khac voi `similarity_threshold`; no la nguong IoU de loai cac box chong len nhau.

---

## 10. Ket Qua Sau Calibration

Ket qua sau calibration:

```text
TEST:
  MAE  : 38.38
  RMSE : 142.79

TEST_COCO:
  MAE  : 38.69
  RMSE : 73.29
```

So voi truoc:

```text
TEST MAE:      50.01 -> 38.38
TEST_COCO MAE: 54.77 -> 38.69
```

Danh gia:

- MAE tot hon ro ret.
- TEST_COCO RMSE tot hon.
- TEST RMSE xau hon, co kha nang do mot so outlier bi sai rat lon.
- Neu bao cao, nen noi calibration cai thien MAE nhung can can nhac outlier/RMSE.

---

## 11. Lenh Chay Neu Da Co PCA Weight

Neu da co:

```text
weights/pca_whitening_best.npz
weights/pca_whitening_best_checkpoint.npz
```

Khong can train lai. Chi can:

```powershell
python calibrate.py
python evaluate.py
```

Tren Kaggle, neu upload code moi kem PCA weight, `kaggle_run.py --mode auto` se bo qua train va chay calibration/evaluate.

---

## 12. Luu Y Khi Nop Bai

Vi thay/co cam neural network, khong nen nop kem file:

```text
weights/siamese_pairwise_MLP_best.pth
```

Code hien tai khong dung file nay, nhung ten file co `MLP` va duoi `.pth`, de gay hieu nham la van dung neural network.

Nen nop cac file classical-only:

```text
src/
calibrate.py
evaluate.py
inference.py
kaggle_run.py
kaggle_install_and_run.sh
requirements.txt
README.md
weights/pca_whitening_best.npz
weights/pca_whitening_best_checkpoint.npz
```

Neu da calibrate xong va muon evaluate lai nhanh:

```text
weights/count_calibration.npz
```

co the nop kem.

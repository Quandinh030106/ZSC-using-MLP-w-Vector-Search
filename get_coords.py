import cv2
import os

IMAGE_PATH = 'data/images_384_VarV2/685.jpg' 

def main():
    if not os.path.exists(IMAGE_PATH):
        print("Không tìm thấy ảnh!")
        return

    img = cv2.imread(IMAGE_PATH)

    # Hiện cửa sổ cho phép dùng chuột kéo thả
    print("1. Dùng chuột TRÁI kéo thả để khoanh vùng vật thể mẫu (con chim, quả dâu...).")
    print("2. Khoanh xong bấm phím SPACE (Cách) hoặc ENTER để xác nhận.")
    print("3. Nếu khoanh sai, bấm chữ 'c' để khoanh lại.")
    
    # Mở cửa sổ ROI (Region of Interest)
    bbox = cv2.selectROI("Chon Vat The Mau (An SPACE de xac nhan)", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # OpenCV trả về tọa độ dạng: (x_min, y_min, width, height)
    x, y, w, h = bbox

    if w > 0 and h > 0:
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)
        
        print("\nTHÀNH CÔNG! Hãy copy dòng dưới đây dán vào file inference.py:\n")
        print(f"EXEMPLAR_BOX = [{xmin}, {ymin}, {xmax}, {ymax}]")
    else:
        print("\nBạn chưa chọn gì cả!")

if __name__ == "__main__":
    main()
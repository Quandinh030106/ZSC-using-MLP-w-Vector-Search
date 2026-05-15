import os

class CFG:
    #  CẤU HÌNH ĐƯỜNG DẪN 
    # Đường dẫn thư mục gốc
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images_384_VarV2')
    JSON_PATH = os.path.join(DATA_DIR, 'annotation_FSC147_384.json')
    SPLIT_PATH = os.path.join(DATA_DIR, 'train_test_val_split.json')
    FEATURES_DIR = os.path.join(DATA_DIR, 'features_v2')
    
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
    BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'siamese_MLP_V2_best.pth')

    #  CẤU HÌNH TRÍCH XUẤT ĐẶC TRƯNG 
    PATCH_SIZE = (64, 64)
    FEATURE_DIM = 1812
    LATENT_DIM = 128

    #  CẤU HÌNH HUẤN LUYỆN 
    BATCH_SIZE = 128
    EPOCHS = 35
    LEARNING_RATE = 0.0005
    MARGIN = 1.0
    MAX_GRAD_NORM = 5.0

    #  CẤU HÌNH INFERENCE
    SIMILARITY_THRESHOLD = 0.98
    NMS_THRESHOLD = 0.25
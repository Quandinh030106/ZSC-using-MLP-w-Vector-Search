import os

def _find_fsc147_data_dir(search_root):
    if not os.path.isdir(search_root):
        return None

    for root, dirs, files in os.walk(search_root):
        if (
            'annotation_FSC147_384.json' in files
            and 'Train_Test_Val_FSC_147.json' in files
            and 'images_384_VarV2' in dirs
        ):
            return root

    return None

def _find_file(search_root, filename):
    if not os.path.isdir(search_root):
        return None

    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)

    return None

class CFG:
    #  CẤU HÌNH ĐƯỜNG DẪN 
    # Đường dẫn thư mục gốc
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    KAGGLE_INPUT_DIR = '/kaggle/input'
    KAGGLE_WORKING_DIR = '/kaggle/working'
    IS_KAGGLE = os.path.isdir(KAGGLE_INPUT_DIR)

    LOCAL_DATA_DIR = r'F:\FSR147\FSC147'
    KAGGLE_DATA_DIR = _find_fsc147_data_dir(KAGGLE_INPUT_DIR)

    DATA_DIR = os.environ.get(
        'ZSC_DATA_DIR',
        KAGGLE_DATA_DIR if IS_KAGGLE and KAGGLE_DATA_DIR is not None else LOCAL_DATA_DIR
    )
    IMAGE_DIR = os.path.join(DATA_DIR, 'images_384_VarV2')
    DENSITY_DIR = os.path.join(DATA_DIR, 'gt_density_map_adaptive_384_VarV2')
    JSON_PATH = os.path.join(DATA_DIR, 'annotation_FSC147_384.json')
    SPLIT_PATH = os.path.join(DATA_DIR, 'Train_Test_Val_FSC_147.json')
    FEATURES_DIR = os.environ.get(
        'ZSC_FEATURES_DIR',
        os.path.join(KAGGLE_WORKING_DIR, 'features_lbp_v3') if IS_KAGGLE else os.path.join(DATA_DIR, 'features_lbp_v3')
    )

    WEIGHTS_DIR = os.environ.get(
        'ZSC_WEIGHTS_DIR',
        os.path.join(KAGGLE_WORKING_DIR, 'weights') if IS_KAGGLE else os.path.join(BASE_DIR, 'weights')
    )
    BEST_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'pca_whitening_best.npz')
    UPLOADED_BEST_MODEL_PATH = _find_file(KAGGLE_INPUT_DIR, 'pca_whitening_best.npz') if IS_KAGGLE else None
    BEST_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, 'pca_whitening_best_checkpoint.npz')
    COUNT_CALIBRATION_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'count_calibration.npz')
    UPLOADED_COUNT_CALIBRATION_PATH = _find_file(KAGGLE_INPUT_DIR, 'count_calibration.npz') if IS_KAGGLE else None

    @staticmethod
    def resolve_best_model_path():
        override_path = os.environ.get('ZSC_MODEL_PATH')
        if override_path:
            return override_path
        if os.path.exists(CFG.BEST_MODEL_SAVE_PATH):
            return CFG.BEST_MODEL_SAVE_PATH
        if CFG.IS_KAGGLE and CFG.UPLOADED_BEST_MODEL_PATH is not None:
            return CFG.UPLOADED_BEST_MODEL_PATH
        return CFG.BEST_MODEL_SAVE_PATH

    @staticmethod
    def resolve_count_calibration_path():
        override_path = os.environ.get('ZSC_COUNT_CALIBRATION_PATH')
        if override_path:
            return override_path
        if os.path.exists(CFG.COUNT_CALIBRATION_SAVE_PATH):
            return CFG.COUNT_CALIBRATION_SAVE_PATH
        if CFG.IS_KAGGLE and CFG.UPLOADED_COUNT_CALIBRATION_PATH is not None:
            return CFG.UPLOADED_COUNT_CALIBRATION_PATH
        return CFG.COUNT_CALIBRATION_SAVE_PATH

    BEST_MODEL_PATH = os.environ.get(
        'ZSC_MODEL_PATH',
        UPLOADED_BEST_MODEL_PATH if IS_KAGGLE and UPLOADED_BEST_MODEL_PATH is not None else BEST_MODEL_SAVE_PATH
    )

    #  CẤU HÌNH TRÍCH XUẤT ĐẶC TRƯNG 
    PATCH_SIZE = (64, 64)
    HOG_DIM = 1764
    HSV_HIST_DIM = 48
    LBP_DIM = 256
    FEATURE_DIM = HOG_DIM + HSV_HIST_DIM + LBP_DIM
    PCA_DIM = 256
    LATENT_DIM = PCA_DIM

    #  CẤU HÌNH PCA/WHITENING
    PCA_DIM_CANDIDATES = [64, 128, 256, 512]
    PCA_MAX_TRAIN_SAMPLES = 50000
    PCA_WHITENING_EPS = 1e-5
    PCA_VALIDATION_MARGIN = 0.10
    RANDOM_SEED = 42

    #  CẤU HÌNH INFERENCE
    WINDOW_SCALES = [0.8, 1.0, 1.2]
    SLIDING_STEP_DIVISOR = 4
    MIN_SLIDING_STEP = 4
    FAISS_TOP_K = 2000
    SIMILARITY_THRESHOLD = 0.35
    NMS_THRESHOLD = 0.25

    #  CẤU HÌNH CALIBRATION ĐẾM
    COUNT_CALIBRATION_MAX_IMAGES = int(os.environ.get('ZSC_COUNT_CALIBRATION_MAX_IMAGES', '300'))
    COUNT_CALIBRATION_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    COUNT_CALIBRATION_NMS_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    COUNT_CALIBRATION_RIDGE = 1e-3

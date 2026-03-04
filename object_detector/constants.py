# This file contains all constants and configuration used in the object detection module

# ImageNet normalization constants (used across datasets and models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Numerical stability constants
EPSILON = 1e-7
MIN_BOX_SIZE = 1e-6

# LW-DETR image sizes (optimized for speed)
LWDETR_IMG_SIZES = {
    "tiny": 384,  # Fastest inference (~2ms)
    "small": 640,  # Good speed-accuracy balance
    "medium": 640,  # Higher accuracy, still fast
}

# Training configuration defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
BASE_BATCH_SIZE = 4  # For learning rate scaling

# Learning rate defaults
DEFAULT_MAX_LR = 1e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_WEIGHT_DECAY = 1e-4

# Optimizer configurations
MUON_DEFAULTS = {
    "lr": 0.02,
    "momentum": 0.95,
}

ADAM_DEFAULTS = {
    "lr": 3e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-10,
}

# Scheduler configurations
DEFAULT_T_0_EPOCHS = 10  # For cosine annealing with warm restarts
DEFAULT_T_MULT = 2
DEFAULT_LR_COMPONENT_DECAY = 0.1  # Decoder LR factor
DEFAULT_LR_VIT_LAYER_DECAY = 0.95  # ViT layer-wise decay
DEFAULT_LR_ENCODER_FACTOR = 0.5  # Encoder LR factor

# IoU thresholds for evaluation
DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DEFAULT_CONFIDENCE_THRESHOLD = 0.0

# Dataset configuration
MAX_ANNOTATION_CACHE_SIZE_MB = 100  # Files larger than this use memory mapping
DEFAULT_UPDATE_INTERVAL = 10  # For progress tracking

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Mathematical constants (for performance optimization)
PI_SQUARED = 3.141592653589793**2
FOUR_OVER_PI_SQUARED = 4.0 / PI_SQUARED
INV_PI_SQUARED = 1.0 / PI_SQUARED

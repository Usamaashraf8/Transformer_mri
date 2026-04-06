import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATASET_DIR, "Training")
TEST_DIR        = os.path.join(DATASET_DIR, "Testing")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASSES         = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES     = len(CLASSES)
IMAGE_SIZE      = 160        # smaller input keeps fine-tuning practical on MPS
VAL_SPLIT       = 0.1        # 10% of training data used for validation

# ── Transformer Backbone ───────────────────────────────────────────────────────
MODEL_NAME              = "swin_t"
USE_PRETRAINED          = True
FREEZE_BACKBONE_EPOCHS  = 2
HEAD_DROPOUT            = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 10
BACKBONE_LR     = 1e-4
HEAD_LR         = 7.5e-4
WEIGHT_DECAY    = 0.05       # AdamW regularisation
NUM_WORKERS     = 0
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM  = 1.0
WARMUP_EPOCHS   = 1

# ── Misc ───────────────────────────────────────────────────────────────────────
SEED            = 42
DEVICE          = "mps"      # auto-fallback handled in code if unavailable

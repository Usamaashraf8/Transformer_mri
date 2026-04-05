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
IMAGE_SIZE      = 224        # resize all images to 224×224
VAL_SPLIT       = 0.1        # 10% of training data used for validation

# ── ViT Architecture (from scratch) ───────────────────────────────────────────
PATCH_SIZE      = 16         # each patch is 16×16 pixels
NUM_PATCHES     = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 196 patches
EMBED_DIM       = 256        # embedding dimension per patch token
NUM_HEADS       = 8          # number of attention heads
DEPTH           = 6          # number of transformer encoder blocks
MLP_DIM         = 1024       # hidden size inside the MLP (feed-forward) block
DROPOUT         = 0.1        # dropout rate throughout the model

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 0.05       # AdamW regularisation
NUM_WORKERS     = 4

# ── Misc ───────────────────────────────────────────────────────────────────────
SEED            = 42
DEVICE          = "mps"      # change to "cuda" if you have a GPU, "cpu" otherwise

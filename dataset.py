import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import config

# ── Transforms ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            config.IMAGE_SIZE,
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    Loads images from a folder structured as:
        root/
            glioma/      *.jpg
            meningioma/  *.jpg
            notumor/     *.jpg
            pituitary/   *.jpg
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []   # list of (image_path, label_index)

        for label_idx, class_name in enumerate(config.CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # full training set (with augmentation)
    full_train = BrainTumorDataset(config.TRAIN_DIR, transform=get_train_transforms())

    # stratified split keeps every class balanced in train and validation.
    grouped_indices = {label_idx: [] for label_idx in range(config.NUM_CLASSES)}
    for idx, (_, label) in enumerate(full_train.samples):
        grouped_indices[label].append(idx)

    train_idx, val_idx = [], []
    rng = random.Random(config.SEED)
    for label_idx in grouped_indices:
        class_indices = grouped_indices[label_idx]
        rng.shuffle(class_indices)
        split = int(len(class_indices) * (1 - config.VAL_SPLIT))
        train_idx.extend(class_indices[:split])
        val_idx.extend(class_indices[split:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # val subset uses no-augmentation transforms
    val_dataset = BrainTumorDataset(config.TRAIN_DIR, transform=get_val_test_transforms())

    pin_memory = torch.cuda.is_available()
    persistent_workers = config.NUM_WORKERS > 0

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        BrainTumorDataset(config.TEST_DIR, transform=get_val_test_transforms()),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"Train samples : {len(train_idx)}")
    print(f"Val   samples : {len(val_idx)}")
    print(f"Test  samples : {len(BrainTumorDataset(config.TEST_DIR))}")
    return train_loader, val_loader, test_loader

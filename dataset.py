import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import config

# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
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

    # split indices into train / val
    n       = len(full_train)
    indices = list(range(n))
    random.shuffle(indices)
    split   = int(n * (1 - config.VAL_SPLIT))
    train_idx, val_idx = indices[:split], indices[split:]

    # val subset uses no-augmentation transforms
    val_dataset = BrainTumorDataset(config.TRAIN_DIR, transform=get_val_test_transforms())

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        BrainTumorDataset(config.TEST_DIR, transform=get_val_test_transforms()),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples : {len(train_idx)}")
    print(f"Val   samples : {len(val_idx)}")
    print(f"Test  samples : {len(BrainTumorDataset(config.TEST_DIR))}")
    return train_loader, val_loader, test_loader

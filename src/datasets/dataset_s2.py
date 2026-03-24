import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import pandas as pd
import ast
import rasterio
from skimage.transform import resize
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import pickle, copy, time
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.loader import load_s2_patch

from config import BEN_CLASSES_19 as CLASSES_19

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_19)}

# STATISTICS
S2_NORMALIZE = transforms.Normalize(
    mean=[
        429.94,
        614.22,
        590.24,
        2218.95,  # 10m: B02,B03,B04,B08
        950.68,
        1792.46,
        2075.47,
        2266.46,
        1594.43,
        1009.33,
    ],  # 20m
    std=[
        572.42,
        582.88,
        675.89,
        1365.46,
        729.90,
        1096.01,
        1273.45,
        1356.14,
        1079.19,
        818.87,
    ],
)


class BigEarthNetS2Dataset(Dataset):
    def __init__(
        self, metadata_df, s2_root, support_transform=None, query_transform=None
    ):
        self.meta = metadata_df.reset_index(drop=True)
        self.s2_root = Path(s2_root)
        self.normalize = S2_NORMALIZE
        self.support_transform = support_transform or S2SupportTransform()
        self.query_transform = query_transform or S2TrainTransform()
        self.class_images = {}
        for idx in range(len(self.meta)):
            cls_int = CLASS_TO_IDX[self.meta.iloc[idx]["primary_label"]]
            self.class_images.setdefault(cls_int, []).append(idx)

    def get_num_classes(self):
        return len(self.class_images)

    def get_class_images(self, cls_idx):
        return self.class_images[cls_idx]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        patch_dir = self.s2_root / row["patch_id"]
        image = load_s2_patch(patch_dir)
        image = self.normalize(image)  # always normalized
        label = CLASS_TO_IDX[row["primary_label"]]
        return image, label, row["patch_id"]


class S2TrainTransform:
    """Augmentations on already-normalized S2 patches."""

    def __call__(self, x):
        if random.random() > 0.5:
            x = TF.hflip(x)
        if random.random() > 0.5:
            x = TF.vflip(x)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        if random.random() > 0.5:
            noise = torch.randn_like(x) * 0.05  # ~50 DN / ~800 std
            x = x + noise
        if random.random() > 0.5:
            c, h, w = x.shape
            erase_h = random.randint(10, 30)
            erase_w = random.randint(10, 30)
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            x[:, top : top + erase_h, left : left + erase_w] = 0.0
        return x


class S2SupportTransform:
    def __call__(self, x):
        return x


class S2ValTransform:
    def __call__(self, x):
        return x

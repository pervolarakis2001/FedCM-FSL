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


from src.utils.loader import load_s1_patch

from config import BEN_CLASSES_19 as CLASSES_19

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_19)}
S1_NORMALIZE = transforms.Normalize(mean=[-12.54, -20.23], std=[5.22, 5.84])


class BigEarthNetS1Dataset(Dataset):
    def __init__(
        self, metadata_df, s1_root, support_transform=None, query_transform=None
    ):
        self.meta = metadata_df.reset_index(drop=True)
        self.s1_root = Path(s1_root)
        self.support_transform = support_transform or S1SupportTransform()
        self.query_transform = query_transform or S1TrainTransform()
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
        patch_dir = self.s1_root / row["s1_name"]
        image = load_s1_patch(patch_dir)  # raw — NO transform here
        label = CLASS_TO_IDX[row["primary_label"]]
        return image, label, row["patch_id"]


class S1TrainTransform:
    """
    Safe augmentations for 2-band S1 patches (2, H, W).
    Rules:
    - Spatial: safe
    - Speckle noise: realistic for SAR — multiplicative noise model
    - NO random erasing: SAR is cloud-transparent so no occlusion simulation
    """

    def __init__(self):
        self.normalize = S1_NORMALIZE

    def __call__(self, x):

        if random.random() > 0.5:
            x = TF.hflip(x)

        if random.random() > 0.5:
            x = TF.vflip(x)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])

        if random.random() > 0.5:
            noise = torch.randn_like(x) * 0.5  # 0.5 dB std — mild speckle
            x = x + noise
        x = self.normalize(x)
        return x


class S1SupportTransform:
    def __init__(self):
        self.normalize = S1_NORMALIZE

    def __call__(self, x):
        if random.random() > 0.5:
            x = TF.hflip(x)
        if random.random() > 0.5:
            x = TF.vflip(x)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        return self.normalize(x)


class S1ValTransform:
    def __call__(self, x):
        return S1_NORMALIZE(x)

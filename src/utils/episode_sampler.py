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


class EpisodeSampler:
    def __init__(
        self, dataset, n_way=5, k_shot=1, q_query=15, seed=None, is_train=True
    ):  # ← add is_train flag
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = np.random.RandomState(seed)
        self.is_train = is_train
        self.class_keys = list(dataset.class_images.keys())
        self.num_classes = len(self.class_keys)

    def sample_episode(self):
        selected = self.rng.choice(self.num_classes, self.n_way, replace=False)
        selected_keys = [self.class_keys[i] for i in selected]

        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []

        for ep_label, cls_idx in enumerate(selected_keys):
            indices = self.dataset.get_class_images(cls_idx)
            if len(indices) < self.k_shot + self.q_query:
                continue
            chosen = self.rng.choice(indices, self.k_shot + self.q_query, replace=False)

            for idx in chosen[: self.k_shot]:
                img, _, _ = self.dataset[idx]
                if self.is_train:
                    img = self.dataset.support_transform(img)  # light
                else:
                    img = self.dataset.query_transform(img)  # val = normalize only
                support_imgs.append(img)
                support_labels.append(ep_label)

            for idx in chosen[self.k_shot :]:
                img, _, _ = self.dataset[idx]
                img = self.dataset.query_transform(img)  # full augment or normalize
                query_imgs.append(img)
                query_labels.append(ep_label)

        if len(support_imgs) < self.n_way * self.k_shot:
            return self.sample_episode()

        return (
            torch.stack(support_imgs),
            torch.tensor(support_labels),
            torch.stack(query_imgs),
            torch.tensor(query_labels),
            torch.tensor(selected_keys),
        )

    def episodes(self, n):
        for _ in range(n):
            yield self.sample_episode()

import torch
import numpy as np
import os
from pathlib import Path
import rasterio
from skimage.transform import resize
import numpy as np
import torch

from config import S1_BANDS, S2_BANDS_10m, S2_BANDS_20m


def load_s2_patch(patch_dir: Path) -> torch.Tensor:
    arrays = []
    for band in S2_BANDS_10m:
        matches = list(patch_dir.glob(f"*_{band}.tif"))
        if not matches:
            # Print exactly what IS in the folder
            all_files = [f.name for f in sorted(patch_dir.iterdir())]
            print(f"✗ Band {band} not found in: {patch_dir.name}")
            print(f"  Files present: {all_files}")
            raise FileNotFoundError(f"Band {band} missing in {patch_dir}")
        with rasterio.open(matches[0]) as src:
            arrays.append(src.read(1).astype(np.float32))

    for band in S2_BANDS_20m:
        matches = list(patch_dir.glob(f"*_{band}.tif"))
        if not matches:
            all_files = [f.name for f in sorted(patch_dir.iterdir())]
            print(f"✗ Band {band} not found in: {patch_dir.name}")
            print(f"  Files present: {all_files}")
            raise FileNotFoundError(f"Band {band} missing in {patch_dir}")
        with rasterio.open(matches[0]) as src:
            arr = src.read(1).astype(np.float32)
            arr = resize(arr / 30000, [120, 120], mode="reflect") * 30000
            arrays.append(arr)

    return torch.from_numpy(np.stack(arrays, axis=0))


def load_s1_patch(patch_dir: Path) -> torch.Tensor:
    arrays = []
    for band in S1_BANDS:
        f = list(patch_dir.glob(f"*_{band}.tif"))[0]
        with rasterio.open(f) as src:
            arrays.append(src.read(1).astype(np.float32))
    return torch.from_numpy(np.stack(arrays, axis=0))

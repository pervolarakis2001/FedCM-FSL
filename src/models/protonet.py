import os
import torch
from torch import nn
import torch.nn.functional as F
import os
from skimage.transform import resize
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F

from src.utils.loader import load_s2_patch


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.1)
        out = self.bn3(self.conv3(out))
        out = F.leaky_relu(out + residual, 0.1)
        return self.drop(self.pool(out))


class ResNet12(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = ResidualBlock(in_channels, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)  # (batch, 512)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, encoder, feat_dim=512, proj_dim=128, use_projection=False):
        super().__init__()
        self.encoder = encoder
        self.out_dim = feat_dim
        self.use_projection = use_projection

        if use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, proj_dim),
            )
            self.proj_dim = proj_dim

    def encode(self, x):
        return self.encoder(x)

    def project(self, features):
        """Map features to shared cross-modal space."""
        return self.projection_head(features)

    def forward(self, support_x, support_y, query_x, n_way):
        s_feat = self.encode(support_x)
        q_feat = self.encode(query_x)

        prototypes = torch.zeros(n_way, self.out_dim, device=s_feat.device)
        for c in range(n_way):
            prototypes[c] = s_feat[support_y == c].mean(dim=0)

        # ||q - p||^2 = ||q||^2 - 2*q.p + ||p||^2
        # q_feat: (Q, D), prototypes: (N, D)
        logits = (
            -(
                torch.sum(q_feat**2, dim=1, keepdim=True)  # (Q, 1)
                - 2 * torch.mm(q_feat, prototypes.T)  # (Q, N)
                + torch.sum(prototypes**2, dim=1, keepdim=True).T  # (1, N)
            )
            / self.out_dim
        )

        return logits, prototypes, s_feat

    def train_episode(
        self,
        s_x,
        s_y,
        q_x,
        q_y,
        n_way,
        method: str = "base",
        true_classes=None,
        global_protos=None,  # for fed_proto and fed_proto_proj
        global_D=None,  # for rpt (KxK global distance matrix)
        obs_mask=None,  # for rpt (KxK observation mask)
        class_to_idx=None,  # maps global class label -> row index in global_D
        lam1=0.1,
    ):
        """
        Local training for each method.

        Args:
            s_x, s_y: Support set images and episode-local labels (0..n_way-1).
            q_x, q_y: Query set images and episode-local labels.
            n_way: Number of classes in the current episode.
            method: One of "base", "fed_proto", "fed_proto_proj", "rpt".
            true_classes: Tensor of shape (n_way,) with global class indices.
            global_protos: Dict {global_class_id: prototype_tensor} from server.
            global_D: Tensor (K, K) global relational distance matrix from server.
            obs_mask: Tensor (K, K) bool mask of observed class pairs.
            class_to_idx: Dict {global_class_id: index_in_global_D}.
            lam1: Weight for regularization loss.
        """
        logits, local_protos, s_feat = self.forward(s_x, s_y, q_x, n_way)
        loss_cls = F.cross_entropy(logits, q_y)
        total_loss = loss_cls

        if method == "base" or true_classes is None:
            acc = (logits.argmax(1) == q_y).float().mean().item() * 100
            return total_loss, acc, local_protos

        # ---- FedProto: MSE in native encoder space ----
        if method == "fed_proto":
            matched_local, matched_global = self._match_protos(
                local_protos, true_classes, global_protos
            )
            if matched_local is not None:
                loss_reg = F.mse_loss(matched_local, matched_global.detach())
                total_loss += lam1 * loss_reg

        # ---- FedProto + Projection: MSE in projected space ----
        elif method == "fed_proto_proj":
            assert self.use_projection, "Enable use_projection=True"
            if global_protos is not None:
                # Project features first, then compute prototypes
                s_proj = self.project(s_feat)
                proj_protos = torch.zeros(n_way, self.proj_dim, device=s_feat.device)
                for c in range(n_way):
                    proj_protos[c] = s_proj[s_y == c].mean(dim=0)

                matched_local, matched_global = self._match_protos(
                    proj_protos, true_classes, global_protos
                )
                if matched_local is not None:
                    loss_reg = F.mse_loss(matched_local, matched_global.detach())
                    total_loss += lam1 * loss_reg

        # ---- RPT: relational distance matrix consistency ----
        elif method == "ours":
            loss_rel = self._relational_loss(
                local_protos, true_classes, global_D, obs_mask, class_to_idx
            )
            if loss_rel is not None:
                total_loss += lam1 * loss_rel

        acc = (logits.argmax(1) == q_y).float().mean().item() * 100
        # Always return native prototypes for analysis
        return total_loss, acc, local_protos

    def _match_protos(self, local_protos, true_classes, global_protos):
        """Match local episode prototypes to global ones by class label."""
        if global_protos is None:
            return None, None

        matched_local = []
        matched_global = []

        for i, true_class in enumerate(true_classes):
            cls = true_class.item() if torch.is_tensor(true_class) else true_class
            if cls not in global_protos:
                continue
            matched_local.append(local_protos[i])
            matched_global.append(global_protos[cls])

        if len(matched_local) < 1:
            return None, None

        return (
            torch.stack(matched_local),
            torch.stack(matched_global).to(local_protos.device),
        )

    def _relational_loss(
        self, local_protos, true_classes, global_D, obs_mask, class_to_idx
    ):
        """
        Compare local inter-class distance structure to global consensus.
        Only computes loss over class pairs that have been observed globally.
        """
        if global_D is None or class_to_idx is None:
            return None

        # Find which episode classes exist in the global matrix
        valid_idx = []  # index in local prototypes
        global_idx = []  # index in global_D

        for i, true_class in enumerate(true_classes):
            cls = true_class.item() if torch.is_tensor(true_class) else true_class
            if cls in class_to_idx:
                valid_idx.append(i)
                global_idx.append(class_to_idx[cls])

        if len(valid_idx) < 2:
            return None

        # Local pairwise distances for valid classes
        valid_protos = local_protos[valid_idx]
        local_D = torch.cdist(
            valid_protos.unsqueeze(0), valid_protos.unsqueeze(0), p=2
        ).squeeze(0)

        # Extract global submatrix
        gi = torch.tensor(global_idx, device=local_protos.device, dtype=torch.long)
        global_D_sub = global_D.to(local_protos.device)[gi][:, gi]

        # Extract observation mask submatrix
        if obs_mask is not None:
            mask_sub = obs_mask.to(local_protos.device)[gi][:, gi]
        else:
            mask_sub = torch.ones_like(global_D_sub, dtype=torch.bool)

        # Zero out diagonal in mask (self-distance is always 0, uninformative)
        mask_sub = mask_sub & ~torch.eye(
            len(gi), dtype=torch.bool, device=mask_sub.device
        )

        # Normalize both matrices independently
        local_max = local_D.max() + 1e-8
        global_max = global_D_sub.max() + 1e-8
        local_D_norm = local_D / local_max
        global_D_norm = global_D_sub / global_max

        # Masked MSE over observed off-diagonal pairs
        diff = (local_D_norm - global_D_norm) ** 2
        loss = (diff * mask_sub.float()).sum() / (mask_sub.float().sum() + 1e-8)

        return loss

    # ------------------------------------------------------------------
    # Communication helpers (called between rounds, not during episodes)
    # ------------------------------------------------------------------

    def get_local_distance_matrix(self, data_loader, device):
        """
        Compute prototypes for ALL local classes using full local data,
        then return the normalized pairwise distance matrix.
        Called at the end of local training before communicating with server.

        Returns:
            D_norm: Tensor (N, N) normalized distance matrix.
            classes: Sorted list of global class labels.
        """
        self.eval()
        class_features = {}

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch[0], batch[1]
                x = x.to(device)
                feat = self.encode(x)
                for i in range(len(y)):
                    cls = y[i].item()
                    if cls not in class_features:
                        class_features[cls] = []
                    class_features[cls].append(feat[i])

        classes = sorted(class_features.keys())
        prototypes = torch.stack(
            [torch.stack(class_features[cls]).mean(dim=0) for cls in classes]
        )

        D = torch.cdist(prototypes.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(
            0
        )
        D_norm = D / (D.max() + 1e-8)

        self.train()
        return D_norm, classes

    def get_projected_prototypes(self, data_loader, device):
        """
        For fed_proto_proj: compute prototypes in projected space using
        full local data. These are sent to the server for aggregation.

        Returns:
            prototypes: Dict {global_class_id: projected_prototype_tensor}.
        """
        assert self.use_projection, "Enable use_projection=True"
        self.eval()
        class_features = {}

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch[0], batch[1]
                x = x.to(device)
                feat = self.encode(x)
                proj = self.project(feat)
                for i in range(len(y)):
                    cls = y[i].item()
                    if cls not in class_features:
                        class_features[cls] = []
                    class_features[cls].append(proj[i])

        prototypes = {}
        for cls in class_features:
            prototypes[cls] = torch.stack(class_features[cls]).mean(dim=0)

        self.train()
        return prototypes

    def get_native_prototypes(self, data_loader, device):
        """
        Compute prototypes in native encoder space using full local data.
        Used by fed_proto for server communication.

        Returns:
            prototypes: Dict {global_class_id: prototype_tensor}.
        """
        self.eval()
        class_features = {}

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch[0], batch[1]
                x = x.to(device)
                feat = self.encode(x)
                for i in range(len(y)):
                    cls = y[i].item()
                    if cls not in class_features:
                        class_features[cls] = []
                    class_features[cls].append(feat[i])

        prototypes = {}
        for cls in class_features:
            prototypes[cls] = torch.stack(class_features[cls]).mean(dim=0)

        self.train()
        return prototypes


# helper wrapper for enabling other benchmark methods
class SplitEncoder(nn.Module):
    def __init__(self, in_channels, shared_body):
        super().__init__()
        # The local stem
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        # The globally shared body
        self.body = shared_body

    def forward(self, x):
        return self.body(self.stem(x))

    # Helper methods so the Client knows exactly what to extract
    def get_shared_weights(self):
        return {k: v.cpu() for k, v in self.body.state_dict().items()}

    def load_shared_weights(self, global_weights):
        self.body.load_state_dict(global_weights)

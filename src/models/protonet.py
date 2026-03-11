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

    def get_shared_weights(self):
        return {k: v.cpu() for k, v in self.body.state_dict().items()}

    def load_shared_weights(self, global_weights):
        self.body.load_state_dict(global_weights)


class ProtoNet(nn.Module):
    def __init__(self, encoder, feat_dim=512):
        super().__init__()
        self.encoder = encoder
        self.out_dim = feat_dim

    def encode(self, x):
        return self.encoder(x)

    def forward(self, support_x, support_y, query_x, n_way):
        s_feat = self.encode(support_x)
        q_feat = self.encode(query_x)
        prototypes = torch.zeros(n_way, self.out_dim, device=s_feat.device)
        for c in range(n_way):
            prototypes[c] = s_feat[support_y == c].mean(dim=0)
        dists = torch.cdist(q_feat, prototypes).pow(2)
        return -dists, prototypes

    # In ProtoNet.train_episode(self):
    def train_episode(
        self, s_x, s_y, q_x, q_y, n_way, true_classes=None, global_protos=None, lam=0.1
    ):
        logits, local_protos = self.forward(s_x, s_y, q_x, n_way)
        loss = F.cross_entropy(logits, q_y)

        if global_protos is not None and lam > 0 and true_classes is not None:
            matched_local = []
            matched_global = []

            for i, true_class in enumerate(true_classes):
                true_class_item = true_class.item()
                if true_class_item not in global_protos:
                    continue
                matched_local.append(local_protos[i])
                matched_global.append(global_protos[true_class_item])

            if len(matched_local) > 0:
                matched_local = torch.stack(matched_local)
                matched_global = torch.stack(matched_global).to(local_protos.device)
                loss_reg = F.mse_loss(matched_local, matched_global)
                loss = loss + lam * loss_reg

        acc = (logits.argmax(1) == q_y).float().mean().item() * 100
        return loss, acc, local_protos

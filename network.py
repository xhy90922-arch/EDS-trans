import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
from timm.models import create_model
import copy
import torch.nn.functional as F
# model.py contains HiFuse_Base / main_model (imported as HFF_model in train scripts)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {
    "vgg11":models.vgg11,
    "vgg13":models.vgg13,
    "vgg16":models.vgg16,
    "vgg19":models.vgg19,
    "vgg11bn":models.vgg11_bn,
    "vgg13bn":models.vgg13_bn,
    "vgg16bn":models.vgg16_bn,
    "vgg19bn":models.vgg19_bn
}


class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

# res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
# "resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}
res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SwinBase(nn.Module):
    def __init__(self, swin_name):
        super(SwinBase, self).__init__()
        swin_dict = {
            "swin_t": models.swin_t(weights="IMAGENET1K_V1"),
            "swin_s": models.swin_s(weights="IMAGENET1K_V1"),
            "swin_v2_t": models.swin_v2_t(weights="IMAGENET1K_V1"),
            "swin_v2_s": models.swin_v2_s(weights="IMAGENET1K_V1"),
            "swin_v2_b": models.swin_v2_b(weights="IMAGENET1K_V1"),
        }
        self.swin_name = swin_name
        if swin_name == "swin_l":

            self.swin = create_model("swin_large_patch4_window7_224", pretrained=True)
            self.pool = self.swin.head.global_pool
        elif swin_name == "swin_b":

            self.swin = create_model("swin_base_patch4_window7_224", pretrained=True)
            self.pool = self.swin.head.global_pool
        else:
            self.swin = copy.deepcopy(swin_dict[swin_name])
        self.in_features = self.swin.head.in_features

        self.swin.head = nn.Sequential()

    def forward(self, x):

        x = self.swin(x)
        if self.swin_name == "swin_l":
            x = self.pool(x)
        elif self.swin_name == "swin_b":
            x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x




class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.)*init_weights)   
    
    def forward(self,x):
        x = self.w*torch.ones((x.shape[0]),1).cuda()
        x = torch.sigmoid(x)
        return x


class source_quantizer(nn.Module):
    """
    Base weight generator Q(·) from Eq.9 in the paper.

    W_q = Q(I_N)

    Takes the identity matrix I_N as input (each row is a one-hot domain indicator)
    and outputs a base weight vector of length N_src.  This provides a domain-level
    prior that is updated every epoch, starting from uniform weights (all sources
    treated equally at initialization).

    Input:  I_N  — identity matrix [N_src, N_src]  (each row = one-hot for one source)
    Output: w    — base weight vector [N_src, 1] → squeezed to [N_src]

    The MLP maps each one-hot row to a scalar logit; these logits are then normalized
    with softmax in the calling code (compute_energy_weights, Eq.11).
    """
    def __init__(self, source_num, hidden_dim=128, type="linear"):
        super(source_quantizer, self).__init__()
        self.type = type
        self.source_num = source_num

        if type == 'wn':
            # Weight-normalized MLP: I_N (one-hot rows) → scalar logit per source
            self.mlp = nn.Sequential(
                weightNorm(nn.Linear(source_num, hidden_dim), name="weight"),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                weightNorm(nn.Linear(hidden_dim, hidden_dim), name="weight"),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                weightNorm(nn.Linear(hidden_dim, 1), name="weight"),
            )
            self.quantizer = self.mlp
            self.quantizer.apply(init_weights)
        else:
            # Simple linear mapping: source_num → 1 scalar per source row
            self.quantizer = nn.Linear(source_num, 1)
            self.quantizer.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: identity matrix I_N [N_src, N_src]
        Returns:
            w: base weight logits [N_src, 1]
               (caller does .unsqueeze(0).squeeze(2) to get [1, N_src])
        """
        x = self.quantizer(x)   # [N_src, N_src] → [N_src, 1]
        return x





class AdaptiveFeatureFusion(nn.Module):
    """
    Fuses Swin Transformer (global) and HWA-local features for use in the bottleneck.
    The HWA dual-scale fusion is done inside main_model (model.py) with a learnable alpha
    parameter (Eq.5 in the paper).  This module projects the HWA class-logits output
    (dim_H = num_classes) into the same space as the Swin feature (dim_F) and combines them,
    providing the bottleneck with a richer combined representation.

    Mapping: combined = alpha * F_norm + (1 - alpha) * proj(H_norm)
    """
    def __init__(self, dim_F, dim_H):
        super(AdaptiveFeatureFusion, self).__init__()
        # alpha starts at 0 (no global bias) and is updated per epoch based on val-accuracy
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.fc_h = nn.Linear(dim_H, dim_F)
        self.offset = nn.Parameter(torch.ones(dim_F))

    def forward(self, features_F, features_H):
        features_F_norm = F.normalize(features_F, p=2, dim=1)
        features_H_norm = F.normalize(features_H, p=2, dim=1)
        features_H_mapped = self.fc_h(features_H_norm)
        alpha = torch.sigmoid(self.alpha)
        combined_features = alpha * features_F_norm + (1 - alpha) * features_H_mapped
        combined_features = combined_features * self.offset
        return combined_features


def compute_energy_score(logits, temperature=1.0):
    """
    Compute temperature-scaled free energy score for each sample (Eq.6-7 in paper).
    E^(j)_i(T) = -T * log( sum_k exp(f^S_{j,k}(x_i) / T) )

    Lower energy => higher compatibility between target sample and source model knowledge.

    Args:
        logits: [B, K] raw logit outputs from a source classifier
        temperature: temperature parameter T controlling distribution smoothness
    Returns:
        energy: [B] energy score per sample (more negative = more compatible)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)




# EDS-Trans

Here is the official implementation of the model **EDS-Trans** (Energy-Aware Dual-Scale Transformer for Multi-Source-Free Domain Adaptation).

## Abstract

Effectively leveraging the rich knowledge embedded in pretrained models has emerged as a critical topic in the era of large models. Multi-Source-Free Domain Adaptation (MSFDA) aims to adaptively transfer knowledge from multiple source-pretrained models to an unlabeled target domain without access to the original source data. However, MSFDA still faces two major challenges: (1) **source-domain imbalance**, where significant heterogeneity in semantic structures and feature distributions across different source domains hinders the model's ability to dynamically adjust source contributions for individual target samples, often leading to negative transfer; and (2) **pseudo-label unreliability**, where domain shift degrades the predictive reliability of source models on the target domain, resulting in confidence-based pseudo-labels containing substantial noise, consequently inducing confirmation bias and degrading performance.

To tackle these challenges, we establish the **energy score** as a unified reliability metric to quantify the distributional compatibility between source models and target samples, thereby guiding source weighting and pseudo-label filtering. Specifically, we propose the **Energy-Aware Dual-Scale Transformer (EDS-Trans)** to alleviate source-domain imbalance, which integrates source knowledge by combining **Hierarchical Window Attention (HWA)** with the **Multi-Source Energy-Aware Dynamic Weighting Strategy**. Additionally, we develop an **Energy-Driven Pseudo-Labeling (EDPL)** strategy that generates initial pseudo labels via dynamically weighted fusion of predictions from multiple source models and leverages joint global and class-wise energy thresholds to filter and refine them. Extensive experiments on four benchmark datasets demonstrate the effectiveness of EDS-Trans compared to the state-of-the-art methods.

## Method

![Framework](image/image2.pdf)

* **EDS-Trans (HWA):** A dual-scale Transformer that jointly encodes global context via the Swin Transformer branch (W-MSA + SW-MSA) and local discriminative features via a shallow small-window self-attention branch (LW-MSA). The two branches are fused with a learnable coefficient α (Eq. 5):
  > F_fu = α · F_S + (1 − α) · F_L

* **Multi-Source Energy-Aware Dynamic Weighting:** Instead of entropy-based confidence, the energy score is used as the reliability metric. The temperature-scaled free energy (Eq. 7) characterizes distributional compatibility between each target sample and each source model. Base weights W_q = Q(I_N) from an MLP quantizer are combined with instance-level energy weights (Eq. 10–11) to produce normalized per-source fusion weights.

* **Energy-Driven Pseudo-Labeling (EDPL):** Initial pseudo labels are generated through dynamically weighted multi-source prediction fusion. Their reliability is evaluated using per-sample energy scores, filtered by joint global threshold τ_g and class-wise threshold τ_c. Low-energy (reliable) samples are used directly for training; high-energy (unreliable) samples are refined via a neighborhood-aware enhancement strategy.

* **Cross-Source Prediction Regularization:** Symmetric KL divergence (Eq. 29) and L2-style matching loss (Eq. 30) are jointly applied to enforce consistency across source model predictions, reducing inter-source disagreement and mitigating negative transfer.

## Setup

### Install Package Dependencies

```
Python Environment: >= 3.7
torch >= 1.10.0
torchvision >= 0.11.0
timm >= 0.5.0
scipy >= 1.5.0
scikit-learn >= 0.24.0
numpy >= 1.19.0
tqdm
argparse, PIL
```

Install dependencies:

```shell
pip install torch torchvision timm scipy scikit-learn numpy tqdm pillow
```

## Datasets

* **Office-31:** Download [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA). Contains 31 classes across 3 domains: Amazon (A), DSLR (D), Webcam (W).
* **Office-Home:** Download [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw). Contains 65 classes across 4 domains: Art (Ar), Clipart (Cl), Product (Pr), Real-World (Re).
* **Office-Caltech:** Download [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) and combine with Office-31. Contains 10 shared classes across 4 domains: Amazon (A), Caltech (C), DSLR (D), Webcam (W).
* **DomainNet:** Download [DomainNet](http://ai.bu.edu/DomainNet/). Contains 345 classes across 6 domains: Clipart (Cl), Infograph (In), Painting (Pa), Quickdraw (Qd), Real (Re), Sketch (Sk).

Place all datasets under `../data/` and generate the `.txt` list files for each domain split.

```
data/
│
├── office-31/
│   ├── amazon/
│   ├── dslr/
│   ├── webcam/
│   ├── amazon_list.txt
│   ├── dslr_list.txt
│   └── webcam_list.txt
│
├── office-home/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   ├── Real_World/
│   ├── Art_list.txt
│   ├── Clipart_list.txt
│   ├── Product_list.txt
│   └── Real_World_list.txt
│
├── office-caltech-10/
│   ├── amazon/
│   ├── caltech/
│   ├── dslr/
│   ├── webcam/
│   └── *_list.txt
│
└── domainnet/
    ├── clipart/
    ├── infograph/
    ├── painting/
    ├── quickdraw/
    ├── real/
    ├── sketch/
    └── *_list.txt
```

## Training

### Step 1: Train Source Models

Train a source model for each source domain independently. The `--s` argument specifies the source domain index and `--t` specifies the target domain index (used only for evaluation during source training).

```shell
# Example: Office-31, source = Amazon (index 0)
python train_source.py --dset office-31 --s 0 --t 1 --max_epoch 100 \
    --trte val --gpu_id 0 --net swin_v2_b --output my/source/

# Example: Office-Home, source = Art (index 0)
python train_source.py --dset office-home --s 0 --t 1 --max_epoch 100 \
    --trte val --gpu_id 0 --net swin_v2_b --output my/source/
```

Repeat for all source domains. Trained checkpoints will be saved to `my/source/{dset}/{domain}/`.

### Step 2: Adapt to Target Domain

Run target-domain adaptation using all available source models. The `--t` argument specifies the target domain index.

```shell
# Example: Office-31, target = DSLR (index 1)
python train_target.py --dset office-31 --t 1 --max_epoch 15 \
    --gpu_id 0 --net swin_b \
    --cls_par 0.7 --crc_par 0.1 --crc_mse 0.1 \
    --temperature 1.15 --beta 0.15 \
    --output_src my/source/ --output my/MSFDA

# Example: Office-Home, target = Clipart (index 1)
python train_target.py --dset office-home --t 1 --max_epoch 15 \
    --gpu_id 0 --net swin_b \
    --cls_par 0.7 --crc_par 0.1 --crc_mse 0.1 \
    --temperature 1.15 --beta 0.15 \
    --output_src my/source/ --output my/MSFDA

# Example: DomainNet, target = Infograph (index 1)
python train_target.py --dset domainnet --t 1 --max_epoch 40 \
    --gpu_id 0 --net swin_b \
    --cls_par 0.7 --crc_par 0.1 --crc_mse 0.1 \
    --temperature 1.15 --beta 0.15 \
    --output_src my/source/ --output my/MSFDA
```


## File Structure

```
EDS-trans/
├── model.py            # HWA dual-scale Transformer (main_model, HFF_block, Global_block, Local_block)
├── network.py          # Backbone networks, feature bottleneck, source_quantizer Q(I_N), AdaptiveFeatureFusion
├── loss.py             # Entropy, KLConsistencyLoss, MSEConsistencyLoss, CrossEntropyLabelSmooth
├── train_source.py     # Source model training script
├── train_target.py     # Target domain adaptation script (EDPL, energy-aware weighting)
├── data_list.py        # Dataset loading utilities
└── swin_transformer.py # Swin Transformer components
```

## Results

EDS-Trans achieves state-of-the-art performance on four benchmark datasets under the MSFDA setting:

| Dataset | Avg. Accuracy |
|---|---|
| Office-31 | **94.0%** |
| Office-Caltech | **98.3%** |
| Office-Home | **89.1%** |
| DomainNet | **63.1%** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{yang2026edstrans,
  title={EDS-Trans: Energy-Aware Dual-Scale Transformer for Multi-Source-Free Domain Adaptation},
  author={Yang, Xinhui and Li, Hongjiao},
  journal={Pattern Analysis and Applications},
  year={2026}
}
```










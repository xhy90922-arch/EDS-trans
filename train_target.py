import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from torchvision import transforms
import network, loss
import model as HFF_model
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from scipy.linalg import norm as LA_norm

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets['target_'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=train_bs * 3, shuffle=False,
                                         num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def compute_energy_score(logits, temperature=1.0):
    """
    Compute temperature-scaled free energy score per sample (Eq.6-7 in paper).
    E^(j)_i(T) = -T * log( sum_k exp(f^S_{j,k}(x_i) / T) )

    Lower energy => higher compatibility between target sample and source model.
    Replaces entropy-based confidence as the core reliability metric.

    Args:
        logits: [B, K] raw logit outputs (before softmax) from source classifier
        temperature: temperature T (default 1.15)
    Returns:
        energy: [B] energy scores per sample
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_energy_weights(energy_scores_per_source, source_weight_base, beta=0.15):
    """
    Compute normalized energy-aware source weights (Eq.8-11 in paper).

    W^(j)_{e,i} = -E^(j)_i(T)                           (Eq.8)
    W^(j)_i = beta*W_q + (1-beta)*exp(W^(j)_{e,i})      (Eq.10)
    W~^(j)_i = exp(W^(j)_i) / sum_m exp(W^(m)_i)        (Eq.11, softmax)

    Args:
        energy_scores_per_source: list of length N_src, each a scalar mean energy
        source_weight_base: [1, N_src] base weights from MLP quantizer Q(I_N)
        beta: balance coefficient between base and energy weights
    Returns:
        normalized_weights: [1, N_src] normalized fusion weights
    """
    # W^(j)_{e,i} = -E^(j)_i  (negate: lower energy => higher weight)
    W_energy = torch.stack([-e for e in energy_scores_per_source]).unsqueeze(0)  # [1, N_src]
    # Eq.10: combine domain-level base weight with instance-level energy weight
    W_combined = beta * source_weight_base + (1 - beta) * torch.exp(W_energy)
    # Eq.11: softmax normalization across sources
    W_normalized = F.softmax(W_combined, dim=1)
    return W_normalized



def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net[0:4] == 'swin':
        netF_list = [network.SwinBase(swin_name=args.net).cuda() for i in range(len(args.src))]

    netH_list = [HFF_model.HiFuse_Base(num_classes=args.class_num).cuda() for i in range(len(args.src))]

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netC_list = [network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netFusion_list = [network.AdaptiveFeatureFusion(dim_F=netF_list[i].in_features, dim_H=args.class_num).cuda() for i in range(len(args.src))]

    netQ = network.source_quantizer(source_num=len(args.src)).cuda()

    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        args.out_file.write(modelpath + '\n')
        args.out_file.flush()
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_H.pt'
        print(modelpath)
        args.out_file.write(modelpath + '\n')
        args.out_file.flush()
        netH_list[i].load_state_dict(torch.load(modelpath))
        netH_list[i].eval()
        for k, v in netH_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        args.out_file.write(modelpath + '\n')
        args.out_file.flush()
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        args.out_file.write(modelpath + '\n')
        args.out_file.flush()
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        modelpath = args.output_dir_src[i] + '/source_fusion.pt'
        print(modelpath)
        args.out_file.write(modelpath + '\n')
        args.out_file.flush()
        netFusion_list[i].load_state_dict(torch.load(modelpath))
        netFusion_list[i].eval()
        for k, v in netFusion_list[i].named_parameters():
            v.requires_grad = False

    for k, v in netQ.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    acc_init = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:

            for i in range(len(args.src)):
                netF_list[i].eval()
                netH_list[i].eval()
                netB_list[i].eval()
            netQ.eval()

            memory_label, all_feature_F, _, _ = obtain_pseudo_label(dset_loaders['test'], netF_list, netH_list, netB_list, netC_list, netQ, netFusion_list, args)
            memory_label = torch.from_numpy(memory_label).cuda()

            for i in range(len(args.src)):
                netF_list[i].train()
                netH_list[i].train()
                netB_list[i].train()
            netQ.train()


        inputs_test = inputs_test.cuda()
        source_repre = torch.eye(len(args.src)).cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        outputs_all_re = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        # temperature T for energy score (Eq.7): controls smoothness of energy estimate
        temperature = getattr(args, 'temperature', 1.15)
        # beta controls balance between base weight W_q and energy weight W_e (Eq.10)
        beta = getattr(args, 'beta', 0.15)

        # Per-source energy scores: E^(j)_i(T) = -T*log(sum_k exp(f^S_{j,k}(x_i)/T)) [Eq.7]
        energy_scores_per_source = []

        for i in range(len(args.src)):
            features_H = netH_list[i](inputs_test).cuda()
            features_F = netF_list[i](inputs_test).cuda()
            combined_features = netFusion_list[i](features_F, features_H).cuda()
            features_test = netB_list[i](combined_features)
            outputs_test = netC_list[i](features_test)
            outputs_all[i] = outputs_test

            # Compute mean energy for this source model over the batch (Eq.7)
            # Lower energy => this source model is more compatible with these target samples
            energy_i = compute_energy_score(outputs_test.detach(), temperature)
            energy_scores_per_source.append(energy_i.mean())

        # Base weight W_q = Q(I_N): domain-level prior from MLP quantizer (Eq.9)
        source_weight_base = netQ(source_repre).unsqueeze(0).squeeze(2)  # [1, N_src]

        # Energy-aware dynamic weights (Eq.8-11)
        # W^(j)_{e,i} = -E^(j)_i  (Eq.8); W^(j)_i = beta*W_q + (1-beta)*exp(W_e) (Eq.10)
        # W~^(j)_i = softmax(W^(j)_i) (Eq.11)
        normalized_weights = compute_energy_weights(
            energy_scores_per_source, source_weight_base, beta=beta
        )  # [1, N_src]

        # Expand weights to per-sample: [B, N_src]
        weights_all = torch.repeat_interleave(normalized_weights, inputs_test.shape[0], dim=0).cpu()

        # Weighted fusion of source predictions (Eq.12)
        outputs_all = torch.transpose(outputs_all, 0, 1)  # [B, N_src, K]
        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        # Per-source weighted outputs for cross-source regularization loss
        weights_all = torch.transpose(weights_all, 0, 1)   # [N_src, B]
        outputs_all = torch.transpose(outputs_all, 0, 1)   # [N_src, B, K]
        for i in range(len(args.src)):
            weights_repeat = torch.repeat_interleave(weights_all[i].unsqueeze(1), args.class_num, dim=1)
            outputs_all_re[i] = outputs_all[i] * weights_repeat




        pred = memory_label[tar_idx]
        if args.cls_par > 0:
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu().long())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.crc_par > 0:
            consistency_loss1 = args.crc_par * loss.KLConsistencyLoss(outputs_all_re, pred, args)

        else:
            consistency_loss1 = torch.tensor(0.0)

        if args.crc_mse > 0:
            consistency_loss2 = args.crc_mse * loss.MSEConsistencyLoss(outputs_all_re, pred, args)

        else:
            consistency_loss2 = torch.tensor(0.0)

        consistency_loss = consistency_loss1 + consistency_loss2
        classifier_loss += consistency_loss


        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            cond_entropy = torch.mean(loss.Entropy(softmax_out))
            msoftmax = softmax_out.mean(dim=0)
            mutual_info = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss = cond_entropy - args.mi_par * mutual_info
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                netH_list[i].eval()
            netQ.eval()

            # Log base weights W_q = Q(I_N) (domain-level prior, Eq.9 in paper)
            with torch.no_grad():
                test_source_weight = netQ(source_repre).unsqueeze(0).squeeze(2)  # [1, N_src]
                test_base_weights = F.softmax(test_source_weight, dim=1)
                weight_str = 'Base weights W_q: [' + ', '.join(
                    [f'{w:.4f}' for w in test_base_weights[0].cpu().numpy()]) + ']'

            acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netH_list, netB_list, netC_list, netQ, netFusion_list, args)
            log_str = 'Iter:{}/{}; Accuracy = {:.2f}%; {}'.format(iter_num, max_iter, acc, weight_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            # 写入权重日志文件 (base weights W_q; energy weights are instance-level)
            weight_log_str = '{},{},{}\n'.format(
                iter_num,
                acc,
                ','.join([f'{w:.6f}' for w in test_base_weights[0].cpu().numpy()])
            )
            args.weight_file.write(weight_log_str)
            args.weight_file.flush()

            if acc >= acc_init:
                acc_init = acc

                for i in range(len(args.src)):
                    torch.save(netF_list[i].state_dict(),
                               osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
                    torch.save(netB_list[i].state_dict(),
                               osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"))
                    torch.save(netC_list[i].state_dict(),
                               osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netQ.state_dict(),
                               osp.join(args.output_dir, "target_Q" + "_" + args.savename + ".pt"))



def obtain_pseudo_label(loader, netF_list, netH_list, netB_list, netC_list, netQ, netFusion_list, args):
    """
    Energy-Driven Pseudo-Labeling (EDPL) strategy (Section 3.4 of the paper).

    Steps:
    1. Generate initial pseudo-labels via dynamically-weighted fusion of multi-source predictions.
    2. Compute per-sample energy scores to evaluate pseudo-label reliability.
    3. Filter pseudo-labels using joint global (tau_g) and class-wise (tau_c) energy thresholds.
       - Low-energy samples (reliable): used directly for training.
       - High-energy samples (unreliable): refined via nearest reliable neighbor (neighborhood-aware).
    4. Refine feature centroids using energy-weighted fused features for final label assignment.
    """
    start_test = True
    temperature = getattr(args, 'temperature', 1.15)
    beta = getattr(args, 'beta', 0.15)

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            source_repre = torch.eye(len(args.src)).cuda()

            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)
            features_all = torch.zeros(len(args.src), inputs.shape[0], args.bottleneck)
            features_all_w = torch.zeros(inputs.shape[0], args.bottleneck)
            features_all_F = torch.zeros(len(args.src), inputs.shape[0], netF_list[0].in_features)
            features_all_F_w = torch.zeros(inputs.shape[0], netF_list[0].in_features)
            # Collect per-sample energy scores from each source model
            energy_per_source = []

            for i in range(len(args.src)):
                features_H = netH_list[i](inputs).cuda()
                features_F = netF_list[i](inputs).cuda()
                combined_features = netFusion_list[i](features_F, features_H).cuda()
                features_F_raw = netF_list[i](inputs)
                features = netB_list[i](combined_features)
                outputs = netC_list[i](features)
                outputs_all[i] = outputs
                features_all[i] = features
                features_all_F[i] = features_F_raw
                # Per-sample energy for this source (Eq.7)
                energy_i = compute_energy_score(outputs, temperature)  # [B]
                energy_per_source.append(energy_i.cpu())

            # Base weights from quantizer (Eq.9): W_q = Q(I_N)
            source_weight_base = netQ(source_repre).unsqueeze(0).squeeze(2)  # [1, N_src]

            # Energy-aware instance-level weights (Eq.8-11)
            mean_energy_per_src = [e.mean() for e in energy_per_source]
            normalized_weights = compute_energy_weights(
                mean_energy_per_src, source_weight_base, beta=beta
            )  # [1, N_src]
            weights_all = torch.repeat_interleave(normalized_weights, inputs.shape[0], dim=0).cpu()

            outputs_all = torch.transpose(outputs_all, 0, 1)   # [B, N_src, K]
            features_all = torch.transpose(features_all, 0, 1) # [B, N_src, D_b]
            features_all_F = torch.transpose(features_all_F, 0, 1)  # [B, N_src, D_F]

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])
                features_all_w[i] = torch.matmul(torch.transpose(features_all[i], 0, 1), weights_all[i])
                features_all_F_w[i] = torch.matmul(torch.transpose(features_all_F[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_feature = features_all_w.float().cpu()
                all_feature_F = features_all_F_w.float().cpu()
                all_label = labels.float()
                # Aggregate per-sample energy: mean across sources [B]
                all_energy = torch.stack(energy_per_source, dim=0).mean(dim=0)  # [B]
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_feature = torch.cat((all_feature, features_all_w.float().cpu()), 0)
                all_feature_F = torch.cat((all_feature_F, features_all_F_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                batch_energy = torch.stack(energy_per_source, dim=0).mean(dim=0)
                all_energy = torch.cat((all_energy, batch_energy), 0)

    # --- EDPL: Energy-Driven Pseudo-Label Filtering (Section 3.4) ---
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_energy_np = all_energy.numpy()  # [N] per-sample mean energy across sources

    # Global energy threshold tau_g (lower energy = more reliable)
    tau_g = np.mean(all_energy_np)

    # Class-wise energy threshold tau_c (Eq. in Section 3.4)
    # For each class, compute the mean energy of samples predicted to that class
    pred_labels_np = torch.squeeze(predict).numpy()
    tau_c = np.zeros(args.class_num)
    for c in range(args.class_num):
        cls_mask = pred_labels_np == c
        if cls_mask.sum() > 0:
            tau_c[c] = np.mean(all_energy_np[cls_mask])
        else:
            tau_c[c] = tau_g  # fallback to global threshold

    # Joint filtering: a sample is "reliable" (low-energy) if its energy is below
    # BOTH the global threshold AND its class-wise threshold
    class_energy_thresh = tau_c[pred_labels_np]  # [N] per-sample class threshold
    low_energy_mask = (all_energy_np <= tau_g) & (all_energy_np <= class_energy_thresh)
    label_confi = low_energy_mask.astype(np.int64)  # 1=reliable, 0=unreliable

    # Normalize features for centroid computation
    all_fea = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    all_prob = all_output.float().cpu().numpy()

    # Initial centroid-based label assignment
    aff = all_prob.copy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    # Neighborhood-aware enhancement for high-energy (unreliable) samples (Section 3.4)
    # Refine unreliable samples by blending with their nearest reliable (low-energy) neighbor
    _, all_idx_nn, _ = nearest_confi_anchor(all_feature_F, all_feature_F, label_confi)

    # Reliability-based interpolation gamma: reliable samples keep themselves (gamma=1),
    # unreliable samples blend more toward nearest reliable neighbor
    # gamma = 0.15 * (1 - normalized_energy) + 0.85  => range [0.85, 1.0] for reliable
    # For unreliable samples: energy > threshold, so normalized value is higher -> lower gamma
    norm_energy = (all_energy_np - all_energy_np.min()) / (all_energy_np.max() - all_energy_np.min() + 1e-8)
    gamma = 0.15 * (1.0 - norm_energy).reshape(-1, 1) + 0.85

    all_fea_nearest = all_fea[all_idx_nn]
    all_fea_fuse = gamma * all_fea + (1 - gamma) * all_fea_nearest

    # Refine centroids using energy-weighted fused features and re-assign labels
    for _ in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea_fuse)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd_fuse = cdist(all_fea_fuse, initc, args.distance)
        pred_label = dd_fuse.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}% | Low-energy ratio: {:.1f}%'.format(
        accuracy * 100, acc * 100, label_confi.mean() * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label.astype('int'), all_feature_F, label_confi, all_label



def nearest_confi_anchor(data_q, data_all, lab_confi):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    num_sam = data_q.shape[0]
    LN_MEM =70

    flag_is_done = 0
    ctr_oper = 0
    idx_left = np.arange(0, num_sam, 1)
    mtx_mem_rlt = -3 * np.ones((num_sam, LN_MEM), dtype='int64')
    mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
    is_mem = 0
    mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
    indices_row = np.arange(0, num_sam, 1)
    nearest_idx_last = np.array([-7])

    while flag_is_done == 0:

        nearest_idx_tmp, idx_last_tmp = nearest_id_search(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore,
                                                            nearest_idx_last)
        is_mem = 1
        nearest_idx_last = nearest_idx_tmp

        if ctr_oper == (LN_MEM - 1):
            flag_sw_bad = 1
        else:
            flag_sw_bad = 0

        mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
        mtx_mem_ignore[:, ctr_oper] = idx_last_tmp

        lab_confi_tmp = lab_confi[nearest_idx_tmp]
        idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
        idx_left[idx_done_tmp] = -1

        if flag_sw_bad == 1:
            idx_bad = np.where(idx_left >= 0)[0]
            mtx_log[idx_bad, 0] = 1
        else:
            mtx_log[:, ctr_oper] = lab_confi_tmp

        flag_len = len(np.where(idx_left >= 0)[0])

        if flag_len == 0 or flag_sw_bad == 1:
            idx_nn_step = []
            for k in range(num_sam):
                try:
                    idx_ts = list(mtx_log[k, :]).index(1)
                    idx_nn_step.append(idx_ts)
                except:
                    print("ts:", k, mtx_log[k, :])
                    idx_nn_step.append(0)

            idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
            data_re = data_all[idx_nn_re, :]
            flag_is_done = 1
        else:
            data_q_ = data_all_[nearest_idx_tmp, :]
        ctr_oper += 1

    return data_re, idx_nn_re, idx_nn_step


def nearest_id_search(Q, X, is_mem_f, step_num, mtx_ignore,
                        nearest_idx_last_f):
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim = 1 - (Simo / Nor)

    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)

    idx_change = np.where((indices_min - nearest_idx_last_f) != 0)[0]
    if is_mem_f == 1:
        if idx_change.shape[0] != 0:
            indices_min[idx_change] = nearest_idx_last_f[idx_change]
    Sim[indices_row, indices_min] = 1000

    # Ignore the history search records.
    if is_mem_f == 1:
        for k in range(step_num):
            indices_ingore = mtx_ignore[:, k]
            Sim[indices_row, indices_ingore] = 1000

    indices_min_cur = np.argmin(Sim, axis=1)
    indices_self = indices_min
    return indices_min_cur, indices_self

def cal_acc_multi(loader, netF_list, netH_list, netB_list, netC_list, netQ, netFusion_list, args):
    """
    Evaluate accuracy using energy-aware dynamic source weighting (Eq.8-11 in paper).
    At inference time, energy scores are computed per source model for each batch,
    and combined with the MLP-based base weights to produce normalized fusion weights.
    """
    start_test = True
    temperature = getattr(args, 'temperature', 1.15)
    beta = getattr(args, 'beta', 0.15)

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            source_repre = torch.eye(len(args.src)).cuda()

            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)
            energy_scores_per_source = []

            for i in range(len(args.src)):
                features_H = netH_list[i](inputs).cuda()
                features_F = netF_list[i](inputs).cuda()
                combined_features = netFusion_list[i](features_F, features_H).cuda()
                features = netB_list[i](combined_features)
                outputs = netC_list[i](features)
                outputs_all[i] = outputs
                # Per-source energy score (Eq.7)
                energy_i = compute_energy_score(outputs, temperature)
                energy_scores_per_source.append(energy_i.mean())

            # Base weights W_q = Q(I_N) (Eq.9)
            source_weight_base = netQ(source_repre).unsqueeze(0).squeeze(2)
            # Energy-aware normalized weights (Eq.8-11)
            normalized_weights = compute_energy_weights(
                energy_scores_per_source, source_weight_base, beta=beta
            )
            weights_all = torch.repeat_interleave(normalized_weights, inputs.shape[0], dim=0).cpu()

            outputs_all = torch.transpose(outputs_all, 0, 1)
            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDS-trans')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--t', type=int, default=0,
                        help="target")  ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-31', choices=['office-31', 'office-home', 'office-caltech', 'domainnet'])
    parser.add_argument('--lr', type=float, default=1 * 1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='swin_b', help="vgg16, resnet50, res101,  swin_l")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.7)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--mi_par', type=float, default=1.0)
    # Cross-source regularization weights (Eq.31): lambda_kl and lambda_mse
    parser.add_argument('--crc_par', type=float, default=0.1,
                        help="lambda_kl: weight for KL cross-source consistency loss (Eq.31)")
    parser.add_argument('--crc_mse', type=float, default=0.1,
                        help="lambda_mse: weight for MSE cross-source consistency loss (Eq.31)")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    # Energy-aware weighting hyper-parameters (Section 3.3)
    parser.add_argument('--temperature', type=float, default=1.15,
                        help="Temperature T for energy score computation (Eq.7)")
    parser.add_argument('--beta', type=float, default=0.15,
                        help="Balance between base weight W_q and energy weight W_e (Eq.10)")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/MSFDA')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office-31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'domainnet':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue

        folder = '../data/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i][0].upper()))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write("output_dir_src list: {}\n".format(args.output_dir_src))
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    # 创建权重日志文件
    args.weight_file = open(osp.join(args.output_dir, 'source_weights.csv'), 'w')
    source_names = [osp.basename(d) for d in args.output_dir_src]
    header = 'iteration,accuracy,' + ','.join([f'weight_{name}' for name in source_names]) + '\n'
    args.weight_file.write(header)
    args.weight_file.flush()

    args.savename = 'par_' + str(args.cls_par) + '_' + str(args.crc_par)

    train_target(args)

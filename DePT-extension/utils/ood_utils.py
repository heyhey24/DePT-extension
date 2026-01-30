# 能量检测模块 (Energy-based OOD Detection)
# 基于 NeurIPS 2020 论文《Energy-based Out-of-distribution Detection》
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import sys
import os
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


recall_level_default = 0.95


def _stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """高精度累积和，并检查最终值与总和是否一致（来自 display_results.py）。"""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            "cumsum was found to be unstable: its last element does not correspond to sum"
        )
    return out


def _fpr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    """在给定召回率下计算 FPR，与 display_results.fpr_and_fdr_at_recall 逻辑一致。"""
    classes = np.unique(y_true)
    if (
        pos_label is None
        and not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        )
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values (descending)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # 找到不同分数的位置，并补上最后一个点
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # 随着阈值降低累积 TP / FP
    tps = _stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - recall_level))

    # FPR = FP / N_neg，其中 N_neg = 总负样本数 = sum(~y_true)
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def _extract_img_from_batch(batch):
    if isinstance(batch, dict):
        if "img" in batch:
            return batch["img"]
        if "image" in batch:
            return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def get_energy_score(model, img, labels=None, temperature: float = 1.0, energy_alpha: float = 0.7):
    """从模型获取图像的能量分数。
    
    参数:
        model: 模型实例（可以是 DataParallel 包装）
        img: 输入图像 tensor
        labels: 标签（可选）
        temperature: 温度系数
        energy_alpha: 双头模型时的能量组合权重
    
    返回:
        energy: 能量分数
    """
    # 处理 DataParallel 包装的模型
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # 确保模型处于评估模式
    model.eval()

    with torch.no_grad():
        # 对于具有显式第二个 head 的模型（ELP 线性探针 / ETFhead），
        # 使用显式的特征和头部接口。
        if (
            hasattr(model_module, "_forward_feats")
            and hasattr(model_module, "_forward_logits_similarity")
            and hasattr(model_module, "_forward_logits_linear_probe")
        ):
            # ELP 系列：线性探针 head
            text_feats, img_feats = model_module._forward_feats(img)

            # 计算相似度 logits
            logits = model_module._forward_logits_similarity(text_feats, img_feats)

            # 计算线性探针 logits（兼容 ETFLinear 等头部实现）
            # _forward_logits_linear_probe 返回 (logits_lp, labels_lp, features_lp)
            logits_lp, _, _ = model_module._forward_logits_linear_probe(
                text_feats, img_feats, labels
            )

            # 计算双头能量
            energy_main = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_lp = -temperature * torch.logsumexp(logits_lp / temperature, dim=1)
            energy = energy_alpha * energy_lp + (1 - energy_alpha) * energy_main

        elif (
            hasattr(model_module, "_forward_feats")
            and hasattr(model_module, "_forward_logits_similarity")
            and hasattr(model_module, "_forward_logits_etf")
        ):
            # ETF 系列：ETF head（如 ETFCoOp / OpenSetETFCoOp）
            text_feats, img_feats = model_module._forward_feats(img)

            # 计算相似度 logits
            logits = model_module._forward_logits_similarity(text_feats, img_feats)

            # 计算 ETF head 的 logits
            # _forward_logits_etf 返回 (logits_etf, labels_etf, features_etf)
            logits_lp, _, _ = model_module._forward_logits_etf(
                text_feats, img_feats, labels
            )

            # 计算双头能量
            energy_main = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_lp = -temperature * torch.logsumexp(logits_lp / temperature, dim=1)
            energy = energy_alpha * energy_lp + (1 - energy_alpha) * energy_main

        else:
            # 对于 CoOp / MaPLe 等单头模型，forward 直接返回最终 logits
            out = model_module(img)
            # 兼容 forward 返回 (loss, logits) 或类似结构的情况
            if isinstance(out, (tuple, list)) and len(out) > 0:
                logits = out[0] if out[0].ndim >= 2 else out[-1]
            else:
                logits = out
            
            # 单头模型：仅计算主 logits 的能量
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)

    return energy


def get_MCM_score(model, img, labels=None, temperature: float = 1.0):
    """从模型获取图像的 MCM (Maximum Concept Matching) 分数。
    
    MCM 分数计算流程：
    1. 获取图像特征和文本特征
    2. 对特征进行归一化
    3. 计算相似度 (image_features @ text_features.T)
    4. 应用 softmax 缩放
    5. 取最大值的负数作为分数
    
    参数:
        model: 模型实例（可以是 DataParallel 包装）
        img: 输入图像 tensor
        labels: 标签（可选）
        temperature: softmax 温度系数
    
    返回:
        mcm_score: MCM 分数（越小越可能是 ID）
    """
    # 处理 DataParallel 包装的模型
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # 确保模型处于评估模式
    model.eval()

    with torch.no_grad():
        # 对于具有 _forward_feats 的模型（ELP / ETF 系列），直接获取特征
        if hasattr(model_module, "_forward_feats"):
            text_feats, img_feats = model_module._forward_feats(img)

            # 归一化图像特征和文本特征
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            # 计算相似度
            output = img_feats @ text_feats.T

        else:
            # 对于 CoOp / MaPLe 等单头模型，需要单独提取特征
            # 提取图像特征
            img_feats = model_module.image_encoder(img.type(model_module.dtype))
            
            # 提取文本特征
            prompts = model_module.prompt_learner()
            tokenized_prompts = model_module.tokenized_prompts
            text_feats = model_module.text_encoder(prompts, tokenized_prompts)
            
            # 归一化图像特征和文本特征
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
            # 计算相似度（不乘 logit_scale，保持原始相似度）
            output = img_feats @ text_feats.T
        
        # 应用 softmax 缩放
        smax = torch.softmax(output / temperature, dim=1)
        
        # MCM 分数：-max(softmax_scores, dim=1)
        mcm_score = -torch.max(smax, dim=1)[0]

    return mcm_score


def calibrate_threshold_from_loader(model, id_val_dataloader, args, percentile: float = 95):
    """基于 ID（base 类）验证集校准 OOD 分数阈值。
    
    参数:
        model: 模型实例
        id_val_dataloader: ID 验证数据的 DataLoader
        args: 参数对象，需包含 score 字段（'energy' 或 'MCM'）
        percentile: 百分位数，默认 95
    
    返回:
        threshold: 计算得到的阈值
    """
    # 获取设备
    if isinstance(model, torch.nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    print(f"[calibrate_threshold_from_loader] 开始校准阈值 (score={args.score}, percentile={percentile})...")

    id_scores = []
    model.eval()
    with torch.no_grad():
        for batch in id_val_dataloader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            # 根据 args.score 选择计算方式
            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"不支持的 score 类型: {args.score}，仅支持 'energy' 或 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            id_scores.append(score_np)

    # 检查是否有数据
    if len(id_scores) == 0:
        raise ValueError("ID 数据加载器为空，无法校准阈值。请检查验证集是否有数据。")

    # 拼接所有分数
    id_scores = np.concatenate(id_scores)

    # 计算阈值（使用百分位）
    threshold = float(np.percentile(id_scores, percentile))

    return threshold


def evaluate_ood_metrics(model, id_data_loader, ood_data_loader, args, threshold=None, log_file: Optional[str] = None):
    """评估 OOD 检测指标。
    
    参数:
        model: 模型实例
        id_data_loader: ID 数据的 DataLoader
        ood_data_loader: OOD 数据的 DataLoader
        args: 参数对象，需包含 score 字段（'energy' 或 'MCM'）
        threshold: OOD 分数阈值（可选，用于计算 OOD 判定比例）
        log_file: 日志文件路径（可选）
    
    返回:
        auroc, aupr, fpr95
    """
    # 获取设备
    if isinstance(model, torch.nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    id_scores = []
    ood_scores = []

    model.eval()

    # 收集 ID 分数
    with torch.no_grad():
        for batch in id_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            # 根据 args.score 选择计算方式
            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"不支持的 score 类型: {args.score}，仅支持 'energy' 或 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            id_scores.append(score_np)

    # 收集 OOD 分数
    with torch.no_grad():
        for batch in ood_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            # 根据 args.score 选择计算方式
            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"不支持的 score 类型: {args.score}，仅支持 'energy' 或 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            ood_scores.append(score_np)

    if len(id_scores) == 0 or len(ood_scores) == 0:
        raise ValueError("ID 或 OOD 数据为空，无法计算 OOD 指标。")

    id_scores = np.concatenate(id_scores)
    ood_scores = np.concatenate(ood_scores)

    # 以 ID 作为正类：
    #  - 正样本（pos）= ID 样本
    #  - 负样本（neg）= OOD 样本
    # 由于分数越大越像 OOD，为了让 "score 越大越像 ID"，这里使用负分数
    pos = (-id_scores).reshape(-1, 1)
    neg = (-ood_scores).reshape(-1, 1)
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1  # 前半段为 ID（正类=1），后半段为 OOD（负类=0）

    # AUROC / AUPR：以 ID 为正类、score 越大越像 ID
    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)

    # FPR@95%TPR：与 _fpr_at_recall 逻辑一致，此时 TPR/Recall 针对的是 ID
    fpr95 = float(_fpr_at_recall(labels, examples, recall_level=recall_level_default))

    # 额外打印基于当前阈值的 OOD 判定比例（与原来逻辑一致）
    if threshold is not None:
        threshold_val = float(threshold)
        flagged = float((ood_scores > threshold_val).sum())
        total_ood = float(len(ood_scores))
        ratio_msg = f"[OOD] ood_train 总数={int(total_ood)}, 判为OOD={int(flagged)}, 比例={flagged/total_ood:.3f}"
        print(ratio_msg)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(ratio_msg + "\n")
            except Exception:
                pass

    summary_msg = f"[OOD] AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR95={fpr95*100:.2f}%"
    print(summary_msg)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(summary_msg + "\n")
        except Exception:
            pass

    return auroc, aupr, fpr95




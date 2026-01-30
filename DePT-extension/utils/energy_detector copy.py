# 能量检测模块 (Energy-based OOD Detection)
# 基于 NeurIPS 2020 论文《Energy-based Out-of-distribution Detection》
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import sys
import os
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score




def _extract_img_from_batch(batch):
    if isinstance(batch, dict):
        if "img" in batch:
            return batch["img"]
        if "image" in batch:
            return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


class EnergyDetector:
    """
    基于能量的 OOD 检测器
    
    功能：
    1. 计算能量分数 E(x) = -T * log(sum(exp(f_i(x)/T)))
    2. 基于验证集确定能量阈值 τ
    3. 判断样本是否为 OOD
    4. 评估 OOD 检测性能（AUROC, FPR95 等）
    """
    
    def __init__(self, temperature=1.0, threshold=None):
        """
        初始化能量检测器
        
        参数:
            temperature (float): 温度系数 T，默认为 1.0
            threshold (float): 能量阈值 τ，若为 None 则需要通过 calibrate() 计算
        """
        self.temperature = temperature
        self.threshold = threshold
        self.energy_stats = {}  # 存储能量统计信息
        
    def compute_energy(self, logits, temperature=None):
        """
        计算能量分数
        
        数学公式：
            E(x; f) = -T * log(sum_i(exp(f_i(x)/T)))
        
        参数:
            logits (Tensor): 模型输出的 logits, shape=[batch_size, num_classes]
            temperature (float): 温度系数，若为 None 则使用 self.temperature
        
        返回:
            energy (Tensor): 能量分数, shape=[batch_size]
                            能量越高表示越可能为 OOD
        """
        if temperature is None:
            temperature = self.temperature
        # -to_np((args.T*torch.logsumexp(output / args.T, dim=1))   
        # E(x) = -T * log(sum(exp(logits/T)))
        # 等价于: E(x) = -T * logsumexp(logits/T)
        # -E(x),正样本就是分布内样本
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy
    
    def combine_energy(self, energy_itm, energy_cat, alpha=0.5):
        """
        组合 ITM 和 CAT 两个 head 的能量分数
        
        参数:
            energy_itm (Tensor): ITM head 的能量分数
            energy_cat (Tensor): CAT head 的能量分数
            alpha (float): 组合权重，E = α*E_ITM + (1-α)*E_CAT
        
        返回:
            combined_energy (Tensor): 组合后的能量分数
        """
        return alpha * energy_cat + (1 - alpha) * energy_itm
    
    def calibrate(self, id_logits_list, percentile=95):
        """
        基于 ID（base 类）验证集的能量分布确定阈值 τ
        
        策略：
        - 使用第 percentile 百分位（默认95%）作为阈值
        - 或使用 μ + 3σ
        
        参数:
            id_logits_list (list of Tensor): ID 样本的 logits 列表
            percentile (float): 百分位数，默认 95（对应 FPR@95）
        
        返回:
            threshold (float): 计算得到的能量阈值 τ
        """
        # 计算所有 ID 样本的能量
        id_energies = []
        for logits in id_logits_list:
            energy = self.compute_energy(logits)
            id_energies.append(energy.cpu().numpy())
        
        id_energies = np.concatenate(id_energies)
        
        # 计算统计信息
        self.energy_stats['mean'] = float(np.mean(id_energies))
        self.energy_stats['std'] = float(np.std(id_energies))
        self.energy_stats['min'] = float(np.min(id_energies))
        self.energy_stats['max'] = float(np.max(id_energies))
        
        # 使用百分位确定阈值
        threshold = float(np.percentile(id_energies, percentile))
        self.threshold = threshold
        
        print(f"[EnergyDetector] 能量统计信息:")
        print(f"  - Mean: {self.energy_stats['mean']:.4f}")
        print(f"  - Std:  {self.energy_stats['std']:.4f}")
        print(f"  - Min:  {self.energy_stats['min']:.4f}")
        print(f"  - Max:  {self.energy_stats['max']:.4f}")
        print(f"  - 阈值 τ (P{percentile}): {threshold:.4f}")
        
        return threshold
    
    def detect_ood(self, logits, temperature=None):
        """
        检测样本是否为 OOD
        
        判断逻辑：
            is_ood = (energy > threshold)
            注意：能量越高越可能为 OOD
        
        参数:
            logits (Tensor): 模型输出的 logits
            temperature (float): 温度系数
        
        返回:
            is_ood (Tensor): 布尔张量，True 表示 OOD
            energy (Tensor): 能量分数
        """
        if self.threshold is None:
            raise ValueError("阈值未设置！请先调用 calibrate() 方法。")
        
        energy = self.compute_energy(logits, temperature)
        is_ood = energy > self.threshold
        
        return is_ood, energy
    
    def save_state(self):
        """保存检测器状态"""
        state = {
            'temperature': self.temperature,
            'threshold': self.threshold,
            'energy_stats': self.energy_stats,
        }
        return state
    
    def load_state(self, state):
        """加载检测器状态"""
        self.temperature = state['temperature']
        self.threshold = state['threshold']
        self.energy_stats = state['energy_stats']


# 全局变量：OOD 检测结果标志（供其他文件引用）
is_ood = None  # 当前批次是否为 OOD 样本的标志（Tensor 或 numpy array）
current_energy = None  # 当前批次的能量分数
ood_detector_instance = None  # OOD 检测器实例


class OODDetectionModule:
    """
    OOD 检测模块 - 实际执行 OOD 检测的主类
    
    功能：
    1. 接收 CustomCLIP 模型和图像输入
    2. 从模型获取 logits 和 logits_lp
    3. 计算能量分数
    4. 判断是否 OOD
    5. 更新全局标志变量 is_ood
    """
    
    def __init__(self, model, temperature=1.0, energy_alpha=0.5, device=None):
        """
        初始化 OOD 检测模块
        
        参数:
            model: CustomCLIP 模型实例
            temperature (float): 温度系数，默认为 1.0
            energy_alpha (float): logits 和 logits_lp 的组合权重，默认 0.5
            device: 计算设备（CPU/GPU）
        """
        self.model = model
        # 处理 DataParallel 包装的模型
        if isinstance(model, torch.nn.DataParallel):
            model_for_device = model.module
        else:
            model_for_device = model

        # 标记模型是否包含显式的第二个 head：
        #  - 线性探针 head（ELP 系列，_forward_logits_linear_probe）
        #  - ETF head（ETFCoOp / OpenSetETFCoOp，_forward_logits_etf）
        self.has_linear_probe = (
            hasattr(model_for_device, "_forward_feats")
            and hasattr(model_for_device, "_forward_logits_similarity")
            and (
                hasattr(model_for_device, "_forward_logits_linear_probe")
                or hasattr(model_for_device, "_forward_logits_etf")
            )
        )

        self.device = device if device is not None else next(model_for_device.parameters()).device
        self.temperature = temperature
        self.energy_alpha = energy_alpha
        self.energy_detector = EnergyDetector(temperature=temperature)
        self.threshold = None
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 设置为全局实例
        global ood_detector_instance
        ood_detector_instance = self
        
        print(f"[OODDetectionModule] 初始化完成 (temperature={temperature}, alpha={energy_alpha})")
    
    def _get_logits_from_model(self, img, labels=None):
        """
        从 CustomCLIP 模型获取 logits 和 logits_lp
        
        参数:
            img: 输入图像 tensor
            labels: 标签（可选）
        
        返回:
            logits: 相似度 logits
            logits_lp: 线性探针 logits
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        # 处理 DataParallel 包装的模型
        if isinstance(self.model, torch.nn.DataParallel):
            model_module = self.model.module
        else:
            model_module = self.model
        
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
                logits_lp, _, _ = model_module._forward_logits_linear_probe(text_feats, img_feats, labels)

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
                logits_lp, _, _ = model_module._forward_logits_etf(text_feats, img_feats, labels)

            else:
                # 对于 CoOp / MaPLe 等单头模型，forward 直接返回最终 logits
                out = model_module(img)
                # 兼容 forward 返回 (loss, logits) 或类似结构的情况
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    logits = out[0] if out[0].ndim >= 2 else out[-1]
                else:
                    logits = out
                # 没有单独的第二个 head 时，直接复用 logits 作为 logits_lp
                logits_lp = logits
        
        return logits, logits_lp
    
    def calibrate_threshold(self, id_data_loader, percentile=95, log_file: Optional[str] = None):
        """
        基于 ID（base 类）验证集校准能量阈值
        
        参数:
            id_data_loader: ID 数据的 DataLoader
            percentile (float): 百分位数，默认 95
        
        返回:
            threshold (float): 计算得到的能量阈值
        """
        msg = f"[OODDetectionModule] 开始校准阈值 (percentile={percentile})..."
        print(msg)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        
        id_logits_list = []
        id_logits_lp_list = [] if self.has_linear_probe else None
        
        self.model.eval()
        with torch.no_grad():
            for batch in id_data_loader:
                img = _extract_img_from_batch(batch)
                
                if isinstance(img, torch.Tensor):
                    img = img.to(self.device)
                
                # 获取 logits
                logits, logits_lp = self._get_logits_from_model(img)
                id_logits_list.append(logits)
                if self.has_linear_probe:
                    id_logits_lp_list.append(logits_lp)
        
        # 检查是否有数据
        if len(id_logits_list) == 0:
            raise ValueError("ID 数据加载器为空，无法校准阈值。请检查验证集是否有数据。")
        
        # 组合 logits 和 logits_lp 计算能量
        id_energies = []
        if self.has_linear_probe:
            # 双头模型：使用主 logits 与 linear probe logits 的组合能量
            for logits, logits_lp in zip(id_logits_list, id_logits_lp_list):
                energy_logits = self.energy_detector.compute_energy(logits)
                energy_logits_lp = self.energy_detector.compute_energy(logits_lp)
                combined_energy = self.energy_detector.combine_energy(
                    energy_logits, energy_logits_lp, self.energy_alpha
                )
                energy_np = combined_energy.cpu().numpy()
                if energy_np.ndim == 0:
                    energy_np = np.array([energy_np])
                elif energy_np.ndim > 1:
                    energy_np = energy_np.flatten()
                id_energies.append(energy_np)
        else:
            # 单头模型（CoOp / MaPLe 等）：仅使用主 logits 的能量
            for logits in id_logits_list:
                energy_logits = self.energy_detector.compute_energy(logits)
                energy_np = energy_logits.cpu().numpy()
                if energy_np.ndim == 0:
                    energy_np = np.array([energy_np])
                elif energy_np.ndim > 1:
                    energy_np = energy_np.flatten()
                id_energies.append(energy_np)
        
        # 检查是否有数据
        if len(id_energies) == 0:
            raise ValueError("ID 数据加载器为空，无法校准阈值。请检查验证集是否有数据。")
        
        # 拼接所有能量分数
        id_energies = np.concatenate(id_energies)
        
        # 计算阈值（使用百分位）
        threshold = float(np.percentile(id_energies, percentile))
        self.threshold = threshold
        self.energy_detector.threshold = threshold
        
        # 保存统计信息
        self.energy_detector.energy_stats['mean'] = float(np.mean(id_energies))
        self.energy_detector.energy_stats['std'] = float(np.std(id_energies))
        self.energy_detector.energy_stats['min'] = float(np.min(id_energies))
        self.energy_detector.energy_stats['max'] = float(np.max(id_energies))
        
        lines = [
            "[OODDetectionModule] 阈值校准完成:",
            f"  - Mean: {self.energy_detector.energy_stats['mean']:.4f}",
            f"  - Std:  {self.energy_detector.energy_stats['std']:.4f}",
            f"  - Min:  {self.energy_detector.energy_stats['min']:.4f}",
            f"  - Max:  {self.energy_detector.energy_stats['max']:.4f}",
            f"  - 阈值 τ (P{percentile}): {threshold:.4f}",
        ]
        for line in lines:
            print(line)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")
            except Exception:
                pass
        
        return threshold
    
    def detect(self, img, labels=None):
        """
        检测输入图像是否为 OOD 样本
        
        参数:
            img: 输入图像 tensor, shape=[batch_size, ...]
            labels: 标签（可选）
            update_global (bool): 是否更新全局标志变量，默认 True
        
        返回:
            is_ood (Tensor): 布尔张量，True 表示 OOD, shape=[batch_size]
            energy (Tensor): 能量分数, shape=[batch_size]
            logits: 相似度 logits
            logits_lp: 线性探针 logits
        """
        if self.threshold is None:
            raise ValueError("阈值未设置！请先调用 calibrate_threshold() 方法。")
        
        # 确保输入在正确的设备上
        if isinstance(img, torch.Tensor):
            img = img.to(self.device)
        
        # 获取 logits
        logits, logits_lp = self._get_logits_from_model(img, labels)

        # 计算能量分数
        energy_logits = self.energy_detector.compute_energy(logits, self.temperature)
        if self.has_linear_probe:
            energy_logits_lp = self.energy_detector.compute_energy(logits_lp, self.temperature)
            # 双头：组合能量
            energy = self.energy_detector.combine_energy(
                energy_logits, energy_logits_lp, self.energy_alpha
            )
        else:
            # 单头：仅使用主 logits 的能量
            energy = energy_logits
        
        # 判断是否 OOD（能量越高越可能为 OOD）
        # 注意：这里是逐样本判断，每个样本独立判断是否为 OOD
        # energy shape: [batch_size]，每个元素对应一个样本的能量分数
        # is_ood_batch shape: [batch_size]，每个元素对应一个样本的 OOD 判断（True/False）
        is_ood_batch = energy > self.threshold
        
        # 确保能量是正确的形状（逐样本比较）
        # energy 应该是1维张量 [batch_size]，即使batch_size=1也应该是shape=[1]而不是标量
        assert energy.dim() == 1, \
            f"能量分数应该是1维张量（每个样本一个分数），但得到 shape={energy.shape}"
        assert len(energy) > 0, "batch大小必须大于0"
        
        return is_ood_batch, energy, logits, logits_lp
    
def evaluate_ood_metrics(detector, id_data_loader, ood_data_loader, log_file: Optional[str] = None):
    id_energies = []
    ood_energies = []

    detector.model.eval()

    # 收集 ID 能量
    with torch.no_grad():
        for batch in id_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(detector.device)

            _, energy, _, _ = detector.detect(img, labels=None)
            energy_np = energy.cpu().numpy()
            if energy_np.ndim == 0:
                energy_np = np.array([energy_np])
            elif energy_np.ndim > 1:
                energy_np = energy_np.flatten()
            id_energies.append(energy_np)

    # 收集 OOD 能量
    with torch.no_grad():
        for batch in ood_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(detector.device)

            _, energy, _, _ = detector.detect(img, labels=None)
            energy_np = energy.cpu().numpy()
            if energy_np.ndim == 0:
                energy_np = np.array([energy_np])
            elif energy_np.ndim > 1:
                energy_np = energy_np.flatten()
            ood_energies.append(energy_np)

    if len(id_energies) == 0 or len(ood_energies) == 0:
        raise ValueError("ID 或 OOD 数据为空，无法计算 OOD 指标。")

    id_energies = np.concatenate(id_energies)
    ood_energies = np.concatenate(ood_energies)

    # 标签：ID=1, OOD=0
    y_true = np.concatenate([
        np.ones_like(id_energies, dtype=np.int32),
        np.zeros_like(ood_energies, dtype=np.int32)
    ])

    # 分数：score 越大越像 ID，这里用 -energy
    scores = -np.concatenate([id_energies, ood_energies])

    # AUROC / AUPR-In
    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    # FPR@95%TPR
    fpr, tpr, _ = roc_curve(y_true, scores)
    target_tpr = 0.95
    idxs = np.where(tpr >= target_tpr)[0]
    if len(idxs) > 0:
        fpr95 = float(fpr[idxs[0]])
    else:
        fpr95 = 1.0

    # 额外打印基于当前阈值的 OOD 判定比例（与原来逻辑一致）
    if detector.threshold is not None:
        threshold = float(detector.threshold)
        flagged = float((ood_energies > threshold).sum())
        total_ood = float(len(ood_energies))
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




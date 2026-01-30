import os
import pickle
import math
import random
import json
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OpenSetOxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        self.cfg = cfg
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED

        # 用数据集目录 + seed + num_shots 构造一个存放待选择的 OOD 类别索引的文件
        self.ood_class_file = os.path.join(
            self.dataset_dir,
            f"openset_ood_class.json",
        )

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        current_stage = cfg.DATASET.CURRENT_STAGE
        
        if subsample == "base":
            #这条分支用于构造类别增量下，baseline的ACC测试集，base类加上ood类样本的测试集
            train, val, test = self.subsample_classes(
                train, val, test, subsample=subsample, stage=current_stage
            )

            super().__init__(train_x=train, val=val, test=test)
        else:
            # 对于 openset 设置，如果是第 0 阶段，直接按 subsample_classes 得到当前阶段的
            # train/val/test，并用其初始化（不做 OOD 拆分）。第0阶段主要构造ood检测的验证集和测试集用于后续评估
            if current_stage == 0:
                _, _, _, ood_test = self._build_stage_split(
                    train, val, test, stage=current_stage + 1
                )
                # 额外公开 OOD 测试数据集供能ood检测等使用
                self.ood_test = ood_test
                train, val, test = self.subsample_classes(
                    train, val, test, subsample=subsample, stage=current_stage
                )
                
                super().__init__(train_x=train, val=val, test=test)
            else:
                # 从第 1 阶段开始，使用 openset 流程：
                # 1) 先根据当前阶段从原始 train 中提取 base 训练集
                (train_base,) = self.subsample_classes(
                    train, subsample=subsample, stage=current_stage
                )

                # 2) 再根据阶段从完整类集合中选出一个 OOD 类，构造 ood_train / 新的 val 与 test
                ood_train, val, test, ood_test = self._build_stage_split(
                    train, val, test, stage=current_stage
                )

                # 额外公开 OOD 训练数据集供ood检测等使用
                self.ood_test = ood_test
                # 先用 train_base 初始化，确保 classnames 只包含 base 类（与训练时的模型匹配）
                # 确保可以正确加载base集上训练的模型，确保维度匹配
                super().__init__(train_x=train_base, val=val, test=test)
                self._train_x = ood_train   #用当前阶段的ood样本进行训练


    def _build_stage_split(self, train, val, test, stage=0):
        """按阶段构造 openset 训练/验证/测试划分，并在首次调用时构造 ood_test_all.

        返回值:
            ood_train: 当前阶段 OOD 训练样本（已重标）
            new_val:   用于阈值校准的验证集（仅 ID，已重标）
            new_test:  当前阶段的测试集（base+历史 OOD+当前 OOD，已重标）
            ood_test:  用于ood检测的测试集
        """

        # 读取配置中的 采样的 OOD 样本数
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            num_ood_samples = 16
        else:
            num_ood_samples = int(getattr(cfg.DATASET, "NUM_OOD_SAMPLES", 16))

        # 当前数据中的所有类别（使用完整 train）
        labels = sorted({item.label for item in train})
        # 构建 label -> classname 的映射，供调试信息使用
        label_to_classname = {}
        for item in train:
            if item.label not in label_to_classname:
                label_to_classname[item.label] = item.classname
        n = len(labels)
        if n == 0:
            return [], val, test

        # 从 openset_ood_class.json 中读取预先定义好的 OOD 类索引列表
        # 假设文件内容是一个整数列表，例如 [10, 3, 25, ...]
        ood_indices = None
        if hasattr(self, "ood_class_file") and os.path.exists(self.ood_class_file):
            try:
                with open(self.ood_class_file, "r") as f:
                    state = json.load(f)
                # 兼容两种格式：
                # 1) 直接是列表
                # 2) 字典中包含 "indices" 或 "ood_indices" 键
                if isinstance(state, list):
                    ood_indices = state
                elif isinstance(state, dict):
                    if "ood_indices" in state:
                        ood_indices = state["ood_indices"]
                    elif "indices" in state:
                        ood_indices = state["indices"]
            except Exception:
                ood_indices = None

        # 如果没有成功读取到 OOD 索引列表，则退化为无 OOD 情况
        if not ood_indices:
            return [], val, test

        # stage 从 1 开始计数：
        #  - 第 stage 个索引对应本阶段训练用的 OOD 类
        #  - 前 stage 个索引用于 nonbase_selected（累计 OOD 类）
        if stage <= 0 or stage > len(ood_indices):
            # 阶段超出预定义范围时，退化为无 OOD
            return [], val, test

        current_ood_index = ood_indices[stage - 1]
        # 当前 OOD 类在全体标签中的具体标签值
        ood_label = current_ood_index

        # 从 train 中抽取该 OOD 类的训练样本（最多 NUM_OOD_SAMPLES 个），稍后用与 val/test 一致的规则重标
        ood_candidates = [it for it in train if it.label == ood_label]
        n_ood = min(num_ood_samples, len(ood_candidates))
        ood_raw = ood_candidates[:n_ood]

        # 按照一半划分 base / OOD 候选：前一半为 base，后一半为 OOD 候选
        m = math.ceil(n / 2)
        base_labels = labels[:m]

        # 使用 base 类 + 累计 OOD 类 构建新的 val/test，并进行标签重标
        # 基础类标签重标为 [0..|base|-1]；被选中的 OOD 类按选择顺序依次映射为
        # len(base_labels)+0, len(base_labels)+1, ...
        base_relabeler = {y: y_new for y_new, y in enumerate(base_labels)}

        # 累计 OOD 类：文件中前 stage 个索引
        # nonbase_selected = []
        # nonbase_selected = ood_indices[:4]
        nonbase_selected = ood_indices[:stage]
        # nonbase_selected.append(ood_label)
        nonbase_relabeler = {
            y: (len(base_labels) + idx) for idx, y in enumerate(nonbase_selected)
        }
        selected_for_eval = set(base_labels) | set(nonbase_selected)

        # 调试信息：打印当前阶段 OOD 类及历史 OOD 类的信息
        # 当前阶段 OOD 类
        current_classname = label_to_classname.get(ood_label, "<unknown>")
        current_remapped = None
        if ood_label in base_relabeler:
            current_remapped = base_relabeler[ood_label]
        elif ood_label in nonbase_relabeler:
            current_remapped = nonbase_relabeler[ood_label]
        print(f"[OpenSetOxfordPets][stage={stage}] current OOD class: "
              f"label={ood_label}, remapped={current_remapped}, classname={current_classname}")

        # 历史所有 OOD 类（包括当前阶段）
        history_info = []
        for y in nonbase_selected:
            cls_name = label_to_classname.get(y, "<unknown>")
            if y in base_relabeler:
                y_new = base_relabeler[y]
            elif y in nonbase_relabeler:
                y_new = nonbase_relabeler[y]
            else:
                y_new = None
            history_info.append(
                {
                    "label": y,
                    "remapped": y_new,
                    "classname": cls_name,
                }
            )
        print(f"[OpenSetOxfordPets][stage={stage}] history OOD classes: {history_info}")

        def _remap(items):
            out = []
            for it in items:
                if it.label not in selected_for_eval:
                    continue
                if it.label in base_relabeler:
                    new_label = base_relabeler[it.label]
                elif it.label in nonbase_relabeler:
                    new_label = nonbase_relabeler[it.label]
                else:
                    continue
                out.append(
                    Datum(
                        impath=it.impath,
                        label=new_label,
                        classname=it.classname,
                    )
                )
            return out

        # 对 ood_train 也进行重标，保证与 val/test 的标签空间一致
        ood_train = _remap(ood_raw)
        new_val = _remap(val)
        new_test = _remap(test)

        # 额外构造一个跨阶段的 OOD 检测测试集 ood_test_all：
        # 包含 base 类测试样本以及待增量的全部的 OOD 类测试样本。
        # 仅在首次可用时构建一次，避免重复计算。
        
        if ood_indices and len(ood_indices) >= 1:
            k = min(5, len(ood_indices))
            nonbase_selected = ood_indices[:k]
            nonbase_relabeler = {
                y: (len(base_labels) + idx) for idx, y in enumerate(nonbase_selected)
            }
            selected_for_eval = set(base_labels) | set(nonbase_selected)
            ood_test = _remap(test)

        return ood_train, new_val, new_test, ood_test


    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    def subsample_classes(self, *args, subsample="base", stage=0):
        """Divide classes into groups.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        # base代表是baseline，这一分支构造的是base类+新增的ood类，用于测试baseline的ACC
        # openset代表是openPT,这一分支构造openPT所要用的数据集
        assert subsample in ["base", "openset"]


        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES! (stage={stage})")

        if subsample == "base":
            # 0 ... m+stage (均为闭区间)，注意 Python 切片右开区间
            selected = labels[:m]

            # 额外：根据 openset_ood_class.json 中前 stage 个索引，
            # 将对应的类别也加入 selected
            if stage > 0 and hasattr(self, "ood_class_file") and os.path.exists(self.ood_class_file):
                try:
                    with open(self.ood_class_file, "r") as f:
                        state = json.load(f)
                    ood_indices = None
                    if isinstance(state, list):
                        ood_indices = state
                    elif isinstance(state, dict):
                        if "ood_indices" in state:
                            ood_indices = state["ood_indices"]
                        elif "indices" in state:
                            ood_indices = state["indices"]
                    if ood_indices:
                        extra = []
                        for idx in ood_indices[:stage]:
                            # 只加入当前数据中存在且尚未被选中的标签
                            if idx in labels and idx not in selected:
                                extra.append(idx)
                        selected = selected + extra
                        print(f"[OpenSetOxfordPets][subsample_classes][stage={stage}] "
                              f"append OOD indices to base selected: {extra}")
                except Exception:
                    # 读取失败则忽略附加 OOD 类
                    pass
        else:  # subsample == "openset"
            # 仅选择索引为 m 的类别（第一个 new 类）作为 openset 类
            selected = labels[:m]

        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
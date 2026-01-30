import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .openset_oxford_pets import OpenSetOxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class OpenSetFood101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        """Open-set variant of Food-101 with staged OOD logic.

        - 使用与原 Food101 相同的方式构建 train/val/test 划分；
        - 复用 OpenSetOxfordPets 中的 openset 流程（subsample_classes + _build_stage_split），
          以避免在多个数据集间复制 openset 逻辑。
        """

        # 1) Food-101 特有的数据读取与 few-shot 预处理
        self.cfg = cfg

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OpenSetOxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OpenSetOxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED

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

        # 2) 复用 OpenSetOxfordPets 的 openset 逻辑
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        current_stage = getattr(cfg.DATASET, "CURRENT_STAGE", 0)

        # 为 Food-101 定义单独的 OOD 类索引文件路径
        self.ood_class_file = os.path.join(
            self.dataset_dir,
            "openset_ood_class.json",
        )

        if subsample == "base":
            # baseline ACC：base 类 + 若干阶段的 OOD 类（与 OpenSetOxfordPets 一致）
            train, val, test = OpenSetOxfordPets.subsample_classes(
                self, train, val, test, subsample=subsample, stage=current_stage
            )
            super().__init__(train_x=train, val=val, test=test)
        else:
            # openset 流程：对 Food-101 复用 OpenSetOxfordPets 的 staged 逻辑
            if current_stage == 0:
                # 阶段 0：仅构造 ood_test，并用当前阶段的 base/openset 划分初始化
                _, _, _, ood_test = OpenSetOxfordPets._build_stage_split(
                    self, train, val, test, stage=current_stage + 1
                )
                self.ood_test = ood_test
                train, val, test = OpenSetOxfordPets.subsample_classes(
                    self, train, val, test, subsample=subsample, stage=current_stage
                )
                super().__init__(train_x=train, val=val, test=test)
            else:
                # 阶段 >=1：先得到当前阶段 base 训练集，再构造 OOD 训练/验证/测试集
                (train_base,) = OpenSetOxfordPets.subsample_classes(
                    self, train, subsample=subsample, stage=current_stage
                )

                ood_train, val, test, ood_test = OpenSetOxfordPets._build_stage_split(
                    self, train, val, test, stage=current_stage
                )

                self.ood_test = ood_test
                # 用 base 训练集初始化，确保类别空间与模型 head 对齐
                super().__init__(train_x=train_base, val=val, test=test)
                # 将 train_x 替换为当前阶段的 OOD 样本用于训练
                self._train_x = ood_train

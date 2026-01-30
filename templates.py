# 0: data root
# 1: seed
# 2: trainer
# 3: dataset
# 4: cfg
# 5: root
# 6: shots
# 7: load epoch
TRAIN_CMD_TEMPLATE_BASE_TO_NEW = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES base '''

TEST_CMD_TEMPLATE_BASE_TO_NEW = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/test_new/{2}/{3}/shots{6}/{4}/seed{1} \
--model-dir {5}/train_base/{2}/{3}/shots{6}/{4}/seed{1} \
--load-epoch {7} \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES new '''

# 0: data root
# 1: seed
# 2: trainer
# 3: dataset
# 4: cfg
# 5: root
# 6: shots
# 7: load dataset
# 8: load epoch
TRAIN_CMD_TEMPLATE_CROSS_DATASET = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/{2}/{3}/shots{6}/{4}/seed{1} \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES all '''

TEST_CMD_TEMPLATE_CROSS_DATASET = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/{2}/{3}/shots{6}/{4}/seed{1} \
--model-dir {5}/{2}/{7}/shots{6}/{4}/seed{1} \
--load-epoch {8} \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES all '''


def get_command(data_root, seed, trainer, dataset, cfg, root, shots, load_dataset, load_epoch, opts=[], mode='b2n', train=True):
    if mode == 'b2n':
        if train:
            cmd = TRAIN_CMD_TEMPLATE_BASE_TO_NEW.format(data_root, seed, trainer, dataset, cfg, root, shots)
        else:
            cmd = TEST_CMD_TEMPLATE_BASE_TO_NEW.format(data_root, seed, trainer, dataset, cfg, root, shots, load_epoch)
    else:
        if train:
            cmd = TRAIN_CMD_TEMPLATE_CROSS_DATASET.format(data_root, seed, trainer, dataset, cfg, root, shots)
        else:
            cmd = TEST_CMD_TEMPLATE_CROSS_DATASET.format(data_root, seed, trainer, dataset, cfg, root, shots, load_dataset, load_epoch)
            
    for opt in opts:
        cmd += f'{opt} '
        
    return cmd

# ======================
# Extra command templates
# ======================

# 0: data root
# 1: seed
# 2: trainer
# 3: dataset
# 4: cfg
# 5: root
# 6: shots
# 7: load epoch (kept for compatibility)
# 8: eval stage (CURRENT_STAGE for data/output)
# 9: base dataset
# 10: base root
TEST_BASELINES_ACC_TEMPLATE = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/acc/{2}/{3}/shots{6}/{4}/stage{8}/seed{1} \
--model-dir {10}/train_base/{2}/{9}/shots{6}/{4}/seed{1} \
--load-epoch -1 \
--eval-only \
DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES base DATASET.CURRENT_STAGE {8} '''


# OOD detection command template
# 0: data root
# 1: seed
# 2: trainer
# 3: dataset (e.g. openset_oxford_pets)
# 4: cfg (e.g. vit_b16_ep10_bs4_lr35)
# 5: output root (used to derive model-dir and output-dir)
# 6: shots
# 7: load epoch
# 8: prev_stage
# 9: warm_start (optional "--model-dir ... --load-epoch ...")
OOD_CMD_TEMPLATE = r'''python utils/ood_detection.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/ood/{3}/shots{6} \
{9}DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES openset DATASET.CURRENT_STAGE {8}'''


def get_ood_command(data_root, seed, trainer, dataset, cfg, root, shots, load_epoch, prev_stage, opts=None):
    """Build a command for utils/ood_detection.py with common defaults.

    与 get_openset_train_command 风格对齐，通过 warm_start 占位符注入
    --model-dir / --load-epoch 参数，而不是事后再 split/替换。
    """
    # 根据阶段构造 warm_start 片段
    base_dataset = dataset.replace('openset_', '', 1)
    base_trainer = trainer.replace('OpenSet', '', 1)
    base_cfg = "vit_b16_ep10_bs4_lr35"
    if prev_stage is not None and load_epoch is not None:
        if prev_stage == 0:
            # 阶段 0：从 base 训练的 ExtrasLinearProbeCoOp checkpoint 热启动
            # 如需泛化到其它数据集，可在此处按 dataset/trainer 做映射
            warm_start = (
                f"--model-dir outputs/coop_dept_etf/train_base/{base_trainer}/{base_dataset}/shots16/vit_b16_ep10_bs4_lr35 \\\n"
                f"--load-epoch {load_epoch} \\\n"
            )
            trainer = base_trainer
            cfg = base_cfg
        else:
            # 阶段 >0：从上一阶段的 openset checkpoint 热启动
            warm_start = (
                f"--model-dir {root}/openset_train/{trainer}/{dataset}/shots{shots}/{cfg}/stage{prev_stage}/seed{seed} \\\n"  # noqa: E501
                f"--load-epoch {load_epoch} \\\n"  # noqa: E501
            )
    else:
        warm_start = ""

    cmd = OOD_CMD_TEMPLATE.format(data_root, seed, trainer, dataset, cfg, root, shots, load_epoch, prev_stage, warm_start,)

    if opts:
        for opt in opts:
            cmd += f" {opt} "
    return cmd

# ==============================
# Open-set staged training template
# ==============================

# OPENSET_TRAIN_CMD_TEMPLATE
# 0: data root
# 1: seed
# 2: trainer (e.g. OpenSetExtrasLinearProbeCoOp)
# 3: dataset (e.g. openset_oxford_pets)
# 4: cfg (e.g. vit_b16_c2_ep10_bs4_lr35)
# 5: output root
# 6: shots
# 7: current stage (CURRENT_STAGE)
# 8: previous stage (for model-dir, -1 or None means no warm start)
# 9: load epoch (for previous stage checkpoint)
OPENSET_TRAIN_CMD_TEMPLATE = r'''python train.py \
--root {0} \
--seed {1} \
--trainer {2} \
--dataset-config-file configs/datasets/{3}.yaml \
--config-file configs/trainers/{2}/{4}.yaml \
--output-dir {5}/openset_train/{2}/{3}/shots{6}/{4}/stage{7}/seed{1} \
{10}DATASET.NUM_SHOTS {6} DATASET.SUBSAMPLE_CLASSES openset DATASET.CURRENT_STAGE {7} '''


def get_openset_train_command(data_root, seed, trainer, dataset, cfg, root, shots, current_stage, prev_stage=None, load_epoch=None, opts=None):
    """Build a staged openset training command (with optional warm start)."""
    base_dataset = dataset.replace('openset_', '', 1)
    if prev_stage is not None and load_epoch is not None:
        if prev_stage == 0:
            # 阶段 0：从 base 训练的 ExtrasLinearProbeCoOp checkpoint 热启动
            base_trainer = "ExtrasLinearProbeCoOp"  # 如果将来有别的 base trainer，可以改成从 cfg 里读
            warm_start = (
                f"--model-dir outputs/coop_dept_etf/train_base/ETFCoOp/{base_dataset}/shots16/vit_b16_ep10_bs4_lr35 \\\n"
                f"--load-epoch {load_epoch} \\\n"
            )
        else:
            # 阶段 >0：从上一阶段的 openset checkpoint 热启动
            warm_start = (
                f"--model-dir {root}/openset_train/{trainer}/{dataset}/shots{shots}/{cfg}/stage{prev_stage}/seed{seed} \\\n"
                f"--load-epoch {load_epoch} \\\n"
            )
    else:
        warm_start = ""

    cmd = OPENSET_TRAIN_CMD_TEMPLATE.format(data_root, seed, trainer, dataset, cfg, root, shots, current_stage, prev_stage, load_epoch, warm_start,)
    
    if opts:
        for opt in opts:
            cmd += f" {opt} "
    return cmd


# zsCLIP evaluation command template（zero-shot eval-only，不依赖 train_base checkpoint）
# 0: data root
# 1: dataset key (e.g. oxford_pets，对应 configs/datasets/{2}.yaml)
# 2: cfg for zsCLIP trainer (e.g. vit_b16_ep10_bs4_lr35)
# 3: output root for zsCLIP
# 4: current_stage
ZSCLIP_EVAL_CMD_TEMPLATE = r'''python train.py \
--root {0} \
--trainer ZeroshotCLIP \
--dataset-config-file configs/datasets/{1}.yaml \
--config-file configs/trainers/CoOp/vit_b16_ep10_bs4_lr35.yaml \
--output-dir {3}/zsclip/{1}/stage{4} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES base DATASET.CURRENT_STAGE {4}'''


def get_zsclip_eval_command(data_root, dataset, cfg, root, stage, opts=None):
    """Build a command for ZeroshotCLIP eval-only on a given dataset."""

    cmd = ZSCLIP_EVAL_CMD_TEMPLATE.format(data_root, dataset, cfg, root, stage,)
    if opts:
        for opt in opts:
            cmd += f" {opt} "
    return cmd
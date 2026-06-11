# 0. 环境配置

    GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

## 0.1 目录作用域

本仓库同时包含“特定机器人本体 / 特定采集格式”的示例代码，以及转换成 LeRobot 后可复用的通用数据结构工具。使用前先按目录判断适用范围：

| 目录 | 适用范围 | 说明 |
|---|---|---|
| [`examples/airbot`](examples/airbot) | **AIRBOT 本体示例** | AIRBOT 的机器人配置、推理、DAgger 采集和 MCAP 转 LeRobot。其它机器人本体应参考该目录另建自己的 `examples/<robot>` 实现。 |
| [`examples/airbot/tools`](examples/airbot/tools) | **AIRBOT + MCAP 专用工具** | 直接读取或改写原始 MCAP 数据，依赖数据目录中的 `config.py`、AIRBOT topic 命名、FlatBuffers 数组编码、`video/mp4` 相机附件等。主要用于转换前检查、预览和修复。 |
| [`scripts`](scripts) | **通用训练 / 标注入口** | 面向 OpenPI/LeRobot 数据结构和训练配置。命令示例里的 `airbot` 配置名只是当前任务示例。 |
| [`scripts/tools`](scripts/tools) | **通用 LeRobot 数据结构工具** | 面向转换后的 LeRobot parquet、`meta/`、fold、VF 标签和模型输出。只要数据集 schema 与脚本需要的列兼容，就可用于其它机器人本体和其它采集格式转换来的数据。 |

后文凡是涉及 **AIRBOT 本体** 或 **MCAP 原始格式** 的步骤都会单独标注；未标注为 MCAP 的 `scripts/tools` 工具默认作用在转换后的 LeRobot 数据集上。


# 1. 数据格式转换与数据工具

## 1.1 AIRBOT / MCAP 示例链路（特定本体 + 特定采集格式）

本节只描述 AIRBOT 采集出的 MCAP 数据如何检查、修复并转换为 LeRobot。其它机器人本体或其它原始数据格式不应直接复用这里的 MCAP 解析逻辑，但转换成 LeRobot 后可以继续使用 §1.2 和后续训练流程。

### 1.1.1 MCAP 转换为 LeRobot（AIRBOT / MCAP 专用）

将 AIRBOT MCAP 采集数据转换为 LeRobot 格式。数据目录下需有 `config.py` 配置文件（定义 `TASK_NAME`、`FOLDERS`、`STATE_TOPICS`、`ACTION_TOPICS`、`CAMERA_TOPICS`、`FPS` 等）。

在 [`scripts/cmds/convert_mcap.sh`](scripts/cmds/convert_mcap.sh) 中修改参数后运行：

    bash scripts/cmds/convert_mcap.sh

底层调用 [`examples/airbot/convert_mcap_data_to_lerobot.py`](examples/airbot/convert_mcap_data_to_lerobot.py)。这个转换器是 AIRBOT/MCAP 专用入口，不是所有本体的通用转换器。

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `DATA_DIR` | `/data/kding/FastWAM/mcap_data/fold_clothv3` | MCAP 数据目录（含 `config.py`） |
| `RESUME` | `true` | `true` = 追加新 episode；`false` = 全量转换 |
| `OVERWRITE` | `false` | `true` = 重新转换覆盖旧数据（仅当 `RESUME=false` 时有效） |
| `SKIP_EPISODES` | `-1` | 仅在 `RESUME=true` 时有效。`-1` = 自动跳过已有 episode 数（适合在**同一** mcap 目录中断恢复）；`0` = 不跳过，从头追加（适合将**新的独立** mcap 目录追加到已有数据集） |
| `INTERVENTION_ONLY` | `false` | `true` = 只截取连续 `intervention=1` 的片段并转成 episode |
| `MIN_INTERVENTION_FRAMES` | `16` | `INTERVENTION_ONLY=true` 时生效，短于该帧数的片段丢弃 |

**两种追加场景说明：**

- **同目录恢复**（中途中断后继续）：`DATA_DIR` 不变，`RESUME=true`，`SKIP_EPISODES=-1`。脚本自动从已有 episode 数处继续，不重复转换。
- **新目录追加**（如将 `fold_clothv2_dagger1` 追加到已由 `fold_clothv2` 生成的数据集）：将 `DATA_DIR` 改为新目录，`RESUME=true`，`SKIP_EPISODES=0`。新目录中的所有文件都会追加进已有数据集，无需将文件手动移入原目录。新目录的 `config.py` 中 `TASK_NAME` 须与原数据集一致。

### 1.1.2 MCAP 原始数据检查与预览（AIRBOT / MCAP 专用）

这些工具都在 [`examples/airbot/tools`](examples/airbot/tools)，作用对象是**转换前的原始 MCAP**，依赖数据目录中的 `config.py` 和 AIRBOT topic/相机附件格式。

| 工具 | 用途 | 示例 |
|---|---|---|
| [`quick_check_mcap.py`](examples/airbot/tools/quick_check_mcap.py) | 快速扫描每个 `.mcap` 的文件大小、视频 FPS、视频帧数和 state/action 消息帧数，定位明显异常 episode。 | `uv run examples/airbot/tools/quick_check_mcap.py --data-dir mcap_data/fold_clothv2` |
| [`visualize_mcap_images.py`](examples/airbot/tools/visualize_mcap_images.py) | 批量生成 episode 预览图，包含 episode 名称、总帧数、时长，以及起始 / 中间 / 结束帧截图。 | `uv run examples/airbot/tools/visualize_mcap_images.py --data-dir mcap_data/fold_clothv2` |
| [`visualize_mcap_web.py`](examples/airbot/tools/visualize_mcap_web.py) | 启动 Web 查看器，浏览、播放、筛选 MCAP episode，并可删除低质量数据。 | `uv run examples/airbot/tools/visualize_mcap_web.py --data-dir mcap_data/fold_clothv2 --port 8765` |
| [`analyze_mcap_timing.py`](examples/airbot/tools/analyze_mcap_timing.py) | 分析帧间 jitter 以及相机视频与 action topic 的时间漂移；支持单文件或整个数据目录聚合。 | `uv run examples/airbot/tools/analyze_mcap_timing.py --data-dir mcap_data/fold_clothv2 --save timing.png` |
| [`mcap_multi_view_video.py`](examples/airbot/tools/mcap_multi_view_video.py) | 从单个 MCAP 的多个 `video/mp4` 相机附件生成横向拼接视频，用于检查多相机同步和内容。 | `uv run examples/airbot/tools/mcap_multi_view_video.py path/to/episode.mcap -o out.mp4` |
| [`visualize_mcap_dataset.ipynb`](examples/airbot/tools/visualize_mcap_dataset.ipynb) | Notebook 版单 episode 检查工具，可列 topic、解码相机流并绘制标量 topic 时间序列。 | 在 Jupyter 中打开并修改 MCAP 路径 |

`visualize_mcap_images.py` 常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | 必填 | MCAP 数据目录（含 `config.py`） |
| `--out-dir` | `<data_dir>/previews` | 预览图输出目录 |
| `--limit` | `0`（全部） | 最多处理的 episode 数，`0` = 不限制 |
| `--fps-threshold` | `15.0` | 低于该 FPS 的视频会在结束时汇总告警 |

`visualize_mcap_web.py` 启动后打开 `http://localhost:8765`（远程开发环境需先做端口转发）。Delete / Backspace 会直接删除磁盘上的当前 MCAP 文件，使用前建议先备份。

### 1.1.3 MCAP 原始数据修复（AIRBOT / MCAP 专用）

这些脚本会改写原始 MCAP，建议先用 `--dry-run` 预览，并确认备份策略。

| 工具 | 用途 | 示例 |
|---|---|---|
| [`fix_mostly_intervention_mcap.py`](examples/airbot/tools/fix_mostly_intervention_mcap.py) | 扫描 `/dagger/intervention`，当某个 episode 中 `intervention=1` 的比例超过阈值时，将该 MCAP 内所有 intervention 消息重写为 0。默认保留 `<file>.mcap.bak`。 | `uv run examples/airbot/tools/fix_mostly_intervention_mcap.py --data_dir mcap_data/fold_clothv2 --threshold 0.8 --dry-run` |
| [`fix_swapped_cameras.py`](examples/airbot/tools/fix_swapped_cameras.py) | 修复指定 folder 范围内 `/env_camera/color/image_raw` 与 `/left_camera/color/image_raw` 相机附件名互换的问题。该脚本只针对已知 AIRBOT 采集事故。 | `uv run examples/airbot/tools/fix_swapped_cameras.py --dat-dir mcap_data/fold_clothv2 --start-folder fold_clothv2_670_680 --end-folder fold_clothv2_700_710 --dry-run` |

## 1.2 通用 LeRobot 数据结构工具（所有本体 / 所有数据格式转换后）

本节工具都作用在**转换后的 LeRobot 数据集**，不关心原始数据来自 MCAP、ROS bag、RLDS 或其它格式。前提是数据已经落到 LeRobot 的 `data/chunk-*/episode_*.parquet` 和 `meta/` 结构，并包含脚本需要的列。

### 1.2.1 增加初始预训练奖励函数真值（通用 LeRobot）

基于成功/失败 episode，为每个时间步计算离散化回报 `binned_value`，保留或补齐 `intervention`，写入阶段信息，并可分配 K-fold。

在 [`scripts/cmds/add_labels.sh`](scripts/cmds/add_labels.sh) 中修改参数后运行：

    bash scripts/cmds/add_labels.sh

底层脚本是 [`scripts/add_returns_to_lerobot.py`](scripts/add_returns_to_lerobot.py) 的 `add_labels` 模式。

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `CONFIG` | 空或 `mcap_data/fold_clothes/config.py` | 可选。只读取 `TASK_NAME`、`FAILED_EPISODES`、`STAGE_BOUNDARIES`；虽然示例来自 MCAP config，但这里读取的是通用标注信息，不读取 MCAP 文件。 |
| `REPO_ID` | `fold_clothv3` | LeRobot 数据集名；留空则从 `CONFIG` 的 `TASK_NAME` 读取 |
| `NUM_FOLDS` | `3` | K-fold 数量；运行后生成 `meta/folds.json` |

底层脚本还支持 `--failed-episodes`、`--stage-boundaries`、`--output-dir`、`--lenient`、`--fold-seed` 等参数，可在 sh 中补充。

### 1.2.2 计算 stats（通用训练数据）

在 [`scripts/cmds/compute_stats.sh`](scripts/cmds/compute_stats.sh) 中修改参数后运行：

    bash scripts/cmds/compute_stats.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `FUNC_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | 训练配置名；当前示例是 AIRBOT 任务配置，其它本体需换成对应配置 |

### 1.2.3 可视化 LeRobot 数据与 VF 标签（通用 LeRobot）

离线可视化单个/多个 episode 的相机采样图、state/action 轨迹、`binned_value` 分布，以及（如果存在）`predicted_value`、`advantage`、`is_good_action`。

在 [`scripts/cmds/visualize_values.sh`](scripts/cmds/visualize_values.sh) 中修改参数后运行：

    bash scripts/cmds/visualize_values.sh

底层脚本是 [`scripts/tools/visualize_values.py`](scripts/tools/visualize_values.py)。

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `DATA_DIR` | `./lerobot_data/Fold_clothes_v3` | LeRobot 数据集目录 |
| `OUTPUT_DIR` | `./assets/visualizations_values/...` | 可视化图片 / 视频输出目录 |
| `EPISODES` | `100,200,300` | 逗号分隔的 episode 编号；留空则处理全部 |
| `NUM_CAMERA_SAMPLES` | `5` | 每个 episode 相机采样帧数 |
| `SKIP_CAMERAS` | `true` | `true` = 跳过相机图像，加快速度 |

### 1.2.4 LeRobot 数据修复与维护工具（通用 LeRobot）

这些工具位于 [`scripts/tools`](scripts/tools)，面向转换后的 LeRobot 数据结构。

| 工具 | 用途 | 示例 |
|---|---|---|
| [`fix_mostly_intervention_lerobot.py`](scripts/tools/fix_mostly_intervention_lerobot.py) | 扫描 episode parquet 的 `intervention` 列；当某个 episode 中 `intervention=1` 的比例超过阈值时，将该列重写为全 0，并同步更新 `meta/episodes_stats.jsonl`。 | `uv run scripts/tools/fix_mostly_intervention_lerobot.py --root lerobot_data/Fold_clothes_v3 --threshold 0.8 --dry-run` |
| [`fix_partial_add_labels.py`](scripts/tools/fix_partial_add_labels.py) | 修复 `add_labels` 中断导致的 parquet 列顺序不一致；同时清理遗留的 `.parquet.tmp`，可选择删除打不开的 parquet。 | `uv run scripts/tools/fix_partial_add_labels.py --repo-id Fold_clothes_v3 --dry-run` |
| [`update_folds_with_new_data.py`](scripts/tools/update_folds_with_new_data.py) | 将新追加但尚未出现在 `meta/folds.json` 的 episode 随机分配到 fold，保留旧 episode 的 fold。 | `uv run scripts/tools/update_folds_with_new_data.py --data-dir lerobot_data/Fold_clothes_v3/data --folds-file lerobot_data/Fold_clothes_v3/meta/folds.json --dry-run` |
| [`visualize_lerobot_dataset.ipynb`](scripts/tools/visualize_lerobot_dataset.ipynb) | Notebook 版 LeRobot episode 检查工具，可看图像、标量特征、summary 和 action/state 对比。 | 在 Jupyter 中打开并修改 `dataset_path` |
| [`visualize_vector_field.py`](scripts/tools/visualize_vector_field.py) | 从 policy checkpoint 可视化指定 episode/timestep 下的扩散 action 矢量场，依赖训练配置和模型 checkpoint。 | 见 [`scripts/cmds/visualize_vector_field.sh`](scripts/cmds/visualize_vector_field.sh) |


# 2. 预训练

本节从 LeRobot 数据集开始，训练 / 标注逻辑总体是通用的；表格中的 `pi06_rl_*_airbot_*` 只是当前 AIRBOT 衣物折叠任务的配置名。其它机器人本体需要在 `src/openpi/training/config.py` 等配置中定义自己的 data/policy/VF 配置。

## 2.1 奖励函数（Value Function）

### 2.1.1 训练奖励函数（单次训练）

在 [`scripts/cmds/vf_train.sh`](scripts/cmds/vf_train.sh) 中修改参数后运行：

    bash scripts/cmds/vf_train.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | 训练配置名 |
| `EXP_NAME` | `vf_v1` | 实验名（checkpoint 子目录） |
| `GPUS` | `0,1,2,3` | 使用的 GPU |
| `NUM_TRAIN_STEPS` | `20000` | 训练步数 |
| `OVERWRITE` | `true` | `false` = resume |

### 2.1.2 K-fold 奖励函数交叉验证训练

单个 VF 在全数据集训练后对全数据集打分会过拟合。K-fold 将数据集分成 K 份，训练 K 个 VF（每个在 K-1 份上训练），每个 VF 只在自己的留出集上打分。

**步骤 1：add_labels 时指定 NUM_FOLDS（见 §1.2.1）**

确保 `scripts/cmds/add_labels.sh` 中 `NUM_FOLDS=3` 已设置，运行后会生成 `meta/folds.json`。

**步骤 2：K-fold 训练（Phase 1）**

在 [`scripts/cmds/vf_kfold_train.sh`](scripts/cmds/vf_kfold_train.sh) 中修改参数后运行：

    bash scripts/cmds/vf_kfold_train.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `REPO_ID` | `fold_clothv3` | LeRobot 数据集 ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | VF 训练配置名 |
| `GPUS` | `(0 1 2 3 4 5)` | 可用 GPU 数组 |
| `NUM_FOLDS` | `3` | K 值 |
| `NUM_TRAIN_STEPS` | `40000` | 每个 VF 的训练步数 |
| `GPUS_PER_FOLD` | `2` | 每个 fold 使用的 GPU 数 |
| `EXP_PREFIX` | `kfold_v3_iter4_wuxi` | K 个 fold checkpoint 的实验名前缀 |
| `RESUME` | `false` | `true` = 从已有 checkpoint 继续 |

**步骤 3：K-fold 推理 + 合并（Phase 2+3）**

在 [`scripts/cmds/vf_kfold_label.sh`](scripts/cmds/vf_kfold_label.sh) 中修改参数后运行：

    bash scripts/cmds/vf_kfold_label.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `REPO_ID` | `fold_clothv3` | LeRobot 数据集 ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | VF 配置名 |
| `GPUS` | `(0 1 2 3 4 5)` | 可用 GPU 数组 |
| `NUM_FOLDS` | `3` | K 值 |
| `GPUS_PER_FOLD` | `2` | 每个 fold 使用的 GPU 数 |
| `EXP_PREFIX` | `kfold_v3_iter4` | 与训练阶段 checkpoint 目录匹配的实验名前缀 |
| `CHECKPOINT_STEP` | `20000` | 留空自动取 fold0 目录下最大数字子目录 |
| `BATCH_SIZE` | `48` | VF 推理批量大小 |
| `POSITIVE_FRACTION` | `0.3` | 正样本比例（预训练 0.3，微调 0.4） |
| `GAMMA` | `0.985` | advantage 折扣因子 |
| `ACTION_HORIZON` | `50` | advantage 累积窗口长度 |

脚本自动完成：
1. 每个 VF 对其留出 fold 推理（values 写入 `VALUES_DIR`）
2. 合并所有 fold 的 values，计算 advantage 和 is_good_action

**也可以手动分步执行：**

    # 训练 fold 0 的 VF（排除 fold 0 的数据）
    CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=./lerobot_data \
    uv run scripts/train.py pi06_rl_vf_airbot_clothes_folding \
        --exp-name kfold_fold0 \
        --data.exclude-fold 0 \
        --overwrite

    # fold 0 的 VF 对 fold 0 推理
    CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=./lerobot_data \
    uv run scripts/add_returns_to_lerobot.py vf_label \
        --repo-id Fold_clothes \
        --vf-config pi06_rl_vf_airbot_clothes_folding \
        --vf-checkpoint-dir checkpoints/pi06_rl_vf_airbot_clothes_folding/kfold_fold0/20000 \
        --infer-fold 0 \
        --values-dir /tmp/vf_kfold_Fold_clothes_0_730

    # ... 对所有 fold 重复 ...

    # 合并所有 fold 的 values
    HF_LEROBOT_HOME=./lerobot_data uv run scripts/add_returns_to_lerobot.py vf_merge \
        --repo-id Fold_clothes \
        --values-dir /tmp/vf_kfold_Fold_clothes \
        --positive-fraction 0.3

### 2.1.3 单 VF 打分（无 K-fold，旧方式）

在 [`scripts/cmds/vf_label.sh`](scripts/cmds/vf_label.sh) 中修改参数后运行：

    bash scripts/cmds/vf_label.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `REPO_ID` | `Fold_clothes` | 数据集 ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | VF 配置名 |
| `VF_CHECKPOINT_DIR` | `checkpoints/.../vf_v1/20000` | checkpoint 路径（含 `params/`） |
| `POSITIVE_FRACTION` | `0.3` | 正样本比例（预训练 0.3，微调 0.4） |
| `BATCH_SIZE` | `32` | 推断批量大小 |


## 2.2 策略训练（π₀.₆*）

### 2.2.1 计算stats

在 [`scripts/cmds/compute_stats.sh`](scripts/cmds/compute_stats.sh) 中修改参数后运行：

    bash scripts/cmds/compute_stats.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `FUNC_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | 训练配置名 |

### 2.2.2 优势条件策略训练

使用 `is_good_action` 标签进行优势条件策略训练，训练时将 `Advantage: Positive/Negative` 注入 prompt，30% 概率 dropout（用于推理时 CFG）。

在 [`scripts/cmds/train_policy.sh`](scripts/cmds/train_policy.sh) 中修改参数后运行：

    bash scripts/cmds/train_policy.sh

| 参数（在 sh 中修改） | 示例值 | 说明 |
|---|---|---|
| `POLICY_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | 训练配置名 |
| `EXP_NAME` | `policy_v3_wospatiodelta_iter4` | 实验名 |
| `GPUS` | `0,1,2,3,4,5,6,7` | 使用的 GPU |
| `OVERWRITE` | `true` | `false` = resume |

- 推理时使用 Classifier-Free Guidance: `ε_guided = ε_uncond + w × (ε_positive − ε_uncond)`，w > 1（论文推荐 w = 2）


# 3. 数据轮转

支持在已标注的 LeRobot 数据集上增量追加新数据，重新标注后继续训练，无需从头重建。
对应论文的迭代循环：收集新数据 → 追加 → 重标注 → K-fold 训练 VF + 重标 is_good_action → 重训 Policy。

VF 标注统一使用 K-fold 交叉验证（见 §2.1.2），避免单 VF 在训练集上打分过拟合。

下面命令以 AIRBOT/MCAP 采集为例。其它本体或其它原始数据格式只需要替换第 1 步的数据转换入口；转换后的 LeRobot 标注、stats、VF、policy 流程仍复用后续步骤。

### 3.1 Iteration 0：初始数据 → 完整 pipeline

    # 1) AIRBOT/MCAP → LeRobot（修改 convert_mcap.sh 中的 DATA_DIR）
    bash scripts/cmds/convert_mcap.sh

    # 2) 添加 binned_value + intervention + fold 分配（修改 add_labels.sh）
    bash scripts/cmds/add_labels.sh

    # 3) 计算 stats
    bash scripts/cmds/compute_stats.sh

    # 4) K-fold 训练 VF（修改 vf_kfold_train.sh 中的 GPUS、EXP_PREFIX 等）
    bash scripts/cmds/vf_kfold_train.sh

    # 5) K-fold 推理 + 合并，写入 is_good_action（修改 vf_kfold_label.sh 中的 POSITIVE_FRACTION 等）
    bash scripts/cmds/vf_kfold_label.sh

    # 6) 训练 Policy（修改 train_policy.sh 中的 EXP_NAME）
    bash scripts/cmds/train_policy.sh

### 3.2 Iteration k：追加新数据 → 重标注 → 重训

以 AIRBOT/MCAP 为例，收集新 MCAP 后有两种方式追加数据（选其一）：

**方式 A：新 MCAP 放入独立目录**（推荐，无需移动文件）
- 将新 MCAP 保存在独立目录（如 `mcap_data/fold_clothv2_dagger1/`），该目录有自己的 `config.py`（`TASK_NAME` 须与已有数据集一致）
- 在 `convert_mcap.sh` 中：`DATA_DIR=mcap_data/fold_clothv2_dagger1`，`RESUME=true`，`SKIP_EPISODES=0`

**方式 B：新 MCAP 合并到原目录**（旧方式）
- 将新 MCAP 文件夹移入原 `mcap_data/fold_clothv2/`，更新其 `config.py`：
  1. 新 MCAP 文件夹加入 `FOLDERS`
  2. 如果成功/失败标签或阶段边界变化，更新 `FAILED_EPISODES` 和 `STAGE_BOUNDARIES`（供 §1.2.1 的 `add_labels` 使用）
  3. DAgger 的逐帧 `intervention` 来自 MCAP 中的 `/dagger/intervention` topic，转换时会写入 LeRobot parquet；通常不需要在 config 中手动列 intervention 片段
- 在 `convert_mcap.sh` 中：`DATA_DIR=mcap_data/fold_clothv2`，`RESUME=true`，`SKIP_EPISODES=-1`

然后修改各 sh 中的相关参数（`EXP_NAME`、`POSITIVE_FRACTION`、`RESUME` 等）后执行：

    # 1) 追加新 episode（按上述方式 A 或 B 设置 convert_mcap.sh）
    bash scripts/cmds/convert_mcap.sh

    # 2) 全量重新标注 binned_value + intervention，重新分配 fold
    bash scripts/cmds/add_labels.sh

    # 3) 重新计算 stats
    bash scripts/cmds/compute_stats.sh

    # 4) K-fold 重训 VF（vf_kfold_train.sh 中 RESUME=false，新建 exp）
    bash scripts/cmds/vf_kfold_train.sh

    # 5) K-fold 推理 + 合并（vf_kfold_label.sh 中 POSITIVE_FRACTION=0.4，微调阶段）
    bash scripts/cmds/vf_kfold_label.sh

    # 6) 重训 Policy（train_policy.sh 中改 EXP_NAME=policy_iter_k）
    bash scripts/cmds/train_policy.sh

    # 7) 部署 policy_iter_k，收集下一轮数据，回到 Step 1


# 4. 部署

当前部署脚本是 AIRBOT 本体示例，入口位于 [`examples/airbot`](examples/airbot)，并通过 [`scripts/cmds/infer_sync.sh`](scripts/cmds/infer_sync.sh) / [`scripts/cmds/infer_async.sh`](scripts/cmds/infer_async.sh) 调用。其它机器人本体需要实现自己的机器人 I/O、同步/异步执行和录制逻辑。

## 4.1 正常部署测试

**Step 1：启动 policy server**

修改 `scripts/cmds/serve_policy.sh` 中的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POLICY_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | policy 配置名 |
| `CHECKPOINT_DIR` | `checkpoints/.../XXXXX` | checkpoint 目录（需填写具体路径） |
| `PORT` | `8000` | 服务端口 |

```bash
bash scripts/cmds/serve_policy.sh
```

**Step 2：运行 AIRBOT 机械臂推理**

有两种推理模式：

**Sync（同步）**：每执行完一个 chunk 才发起下一次推理。延迟较高，适合快速验证。

修改 `scripts/cmds/infer_sync.sh` 中的参数后运行：

```bash
bash scripts/cmds/infer_sync.sh
```

**Async（异步）**：推理与执行并行，支持 TCS 时序平滑，实时性更好，推荐正式部署使用。

修改 `scripts/cmds/infer_async.sh` 中的参数后运行：

```bash
bash scripts/cmds/infer_async.sh
```

两个推理脚本的公共参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `127.0.0.1` | policy server 地址 |
| `PORT` | `8000` | policy server 端口 |
| `PROMPT` | `"Fold clothes"` | 任务描述文本 |
| `CHUNK_SIZE_EXECUTE` | `25` | 每次执行的 action chunk 长度 |
| `RECORD` | `false` | 是否保存 MCAP 录制数据 |
| `RECORD_DIR` | `./inference_data` | MCAP 保存目录 |
| `DAGGER` | `false` | 是否启用 DAgger 干预采集 |

Async 专有参数（TCS 时序 chunk 平滑）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TCS_DROP_MAX` | `12` | 推理延迟补偿：新 chunk 到达时丢弃已过期的前 N 步（N = min(实际延迟步数, tcs_drop_max)），避免执行过时动作 |
| `TCS_MIN_OVERLAP` | `8` | 新旧 chunk 混合时的最小重叠窗口长度；重叠区内线性权重从旧→新渐变，消除动作跳变 |
| `INITIAL_ACTION_WAIT_S` | `10.0` | 首帧启动等待时限（秒）：episode 开始时等待第一个 action chunk 的最长时间，超时则保持当前姿态直到推理就绪 |

键盘控制：
- `Enter` — 开始新 episode
- `R` — 重置当前 episode
- `Q` — 退出

## 4.2 DAgger 在线干预 RL 数据集采集（AIRBOT / MCAP 专用）

DAgger (Dataset Aggregation) 允许在策略推理过程中实时切换到人类遥操作模式，采集纠正数据用于迭代训练。

### 4.2.1 原理

四状态状态机：

    INFERENCE → (按 'i') → ALIGNING → (对齐完成) → DEMONSTRATING → (按 'o') → RESUMING → INFERENCE

- **INFERENCE**：策略正常推理，action 由模型生成（intervention=0）
- **ALIGNING**：主臂通过余弦插值平滑移动到从臂当前位置，防止突然跳变
- **DEMONSTRATING**：人类操作主臂，从臂跟随，采集人类数据（intervention=1）
- **RESUMING**：后台线程归位主臂，重置 action chunk 索引，恢复推理

### 4.2.2 使用方法

1. 启动 policy server（同 4.1 Step 1）

2. 在 `scripts/cmds/infer_sync.sh`（或 `infer_async.sh`）中设置 `DAGGER=true`、`RECORD=true`、`RECORD_DIR=./dagger_data`，然后运行：

```bash
bash scripts/cmds/infer_sync.sh
```

3. 键盘控制：
   - `Enter` — 开始新 episode
   - `i` — 进入人类干预模式（主臂对齐后可遥操作）
   - `o` — 恢复策略推理
   - `q` — 退出

### 4.2.3 数据录制

启用 `RECORD=true` 后，每个 episode 自动保存为 MCAP 文件，格式兼容 [`examples/airbot/convert_mcap_data_to_lerobot.py`](examples/airbot/convert_mcap_data_to_lerobot.py) 转换脚本。录制内容包括：

- 关节状态（follow + lead topics）
- 相机图像（H264 编码）
- 干预标记（`/dagger/intervention` 通道：0=策略，1=人类）

### 4.2.4 DAgger 进阶参数

在推理脚本的 CONFIG 块中可调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DAGGER` | `false` | 是否启用 DAgger 模式 |
| `dagger.key_enter_dagger` | `i` | 进入干预的按键（在脚本中通过 `--dagger.key-enter-dagger` 传递） |
| `dagger.key_resume_inference` | `o` | 恢复推理的按键 |
| `dagger.align_steps` | `50` | 对齐插值步数 |
| `dagger.align_duration` | `1.0` | 对齐总时长（秒） |
| `RECORD_DIR` | `./inference_data` | MCAP 文件保存目录 |

### 4.2.5 硬件要求

DAgger 模式需要主臂（leader）连接。在 `robot_config.py` 中配置：

    robot_groups: ["left", "right"]
    robot_ports: [50051, 50053]      # 从臂 gRPC 端口
    leader_ports: [50050, 50052]     # 主臂 gRPC 端口

### 4.2.6 数据轮转集成

DAgger 采集的数据可直接进入 §3 的迭代训练流程：

```bash
# 1) 在 scripts/cmds/convert_mcap.sh 中设置 DATA_DIR=dagger_data，RESUME=true，然后：
bash scripts/cmds/convert_mcap.sh

# 2) 如有新的失败 episode 或阶段边界，更新 config.py 中的 FAILED_EPISODES / STAGE_BOUNDARIES
#    DAgger intervention=1 的逐帧标签会从 MCAP topic 自动进入 LeRobot parquet
# 3) 继续 §3.2 的重标注 → 重训流程
```

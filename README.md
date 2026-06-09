# 0. 环境配置

    GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .


# 1. 数据格式转换

## 1.1 Airbot格式转换

### 1.1.1 mcap转换为lerobot数据

将 mcap 采集数据转换为 LeRobot 格式。数据目录下需有 `config.py` 配置文件（定义 FOLDERS、TOPICS、FPS 等）。

在 [`scripts/cmds/convert_mcap.sh`](scripts/cmds/convert_mcap.sh) 中修改参数后运行：

    bash scripts/cmds/convert_mcap.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `mcap_data/fold_clothes` | mcap 数据目录（含 config.py） |
| `RESUME` | `false` | `true` = 追加新 episode；`false` = 全量转换 |
| `OVERWRITE` | `true` | `true` = 重新转换覆盖旧数据（仅当 `RESUME=false` 时有效） |
| `SKIP_EPISODES` | `-1` | 仅在 `RESUME=true` 时有效。`-1` = 自动跳过已有 episode 数（适合在**同一** mcap 目录中断恢复）；`0` = 不跳过，从头追加（适合将**新的独立** mcap 目录追加到已有数据集） |

**两种追加场景说明：**

- **同目录恢复**（中途中断后继续）：`DATA_DIR` 不变，`RESUME=true`，`SKIP_EPISODES=-1`。脚本自动从已有 episode 数处继续，不重复转换。
- **新目录追加**（如将 `fold_clothv2_dagger1` 追加到已由 `fold_clothv2` 生成的数据集）：将 `DATA_DIR` 改为新目录，`RESUME=true`，`SKIP_EPISODES=0`。新目录中的所有文件都会追加进已有数据集，无需将文件手动移入原目录。新目录的 `config.py` 中 `TASK_NAME` 须与原数据集一致。

### 1.1.2 增加初始预训练奖励函数真值（模式1：add_labels）

基于二值成功/失败标签，为每个时间步计算离散化回报 `binned_value` 和干预标签 `intervention`，并分配 K-fold。

在 [`scripts/cmds/add_labels.sh`](scripts/cmds/add_labels.sh) 中修改参数后运行：

    bash scripts/cmds/add_labels.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `CONFIG` | `mcap_data/fold_clothes/config.py` | 数据目录下的 config.py |
| `REPO_ID` | `Fold_clothes` | 数据集名，留空则从 config 的 `TASK_NAME` 读取 |
| `NUM_FOLDS` | `3` | K-fold 数量 |

底层脚本还支持 `--failed-episodes`、`--intervention-episodes`、`--output-dir`、`--lenient` 等参数，可在 sh 中补充。

### 1.1.3 计算stats

在 [`scripts/cmds/compute_stats.sh`](scripts/cmds/compute_stats.sh) 中修改参数后运行：

    bash scripts/cmds/compute_stats.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `FUNC_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | 训练配置名 |

### 1.1.4 MCAP 原始数据预览（visualize_mcap_images）

在转换前批量生成 MCAP episode 的预览图，每张图包含：episode 名称、总帧数、时长，以及起始 / 中间 / 结束三帧截图。适合快速检查采集质量。

    uv run scripts/tools/visualize_mcap_images.py --data-dir mcap_data/fold_clothv2

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | 必填 | mcap 数据目录（含 `config.py`） |
| `--out-dir` | `<data_dir>/previews` | 预览图输出目录 |
| `--limit` | `0`（全部） | 最多处理的 episode 数，`0` = 不限制 |

输出结构与数据目录保持一致：`<out_dir>/<folder>/<episode>.jpg`。

---

### 1.1.5 MCAP 交互式 Web 查看器（visualize_mcap_web）

启动一个本地 Web 服务，在浏览器中浏览、播放、筛选 MCAP episode，并可一键删除低质量数据。适合在转换前清理数据集。

    uv run scripts/tools/visualize_mcap_web.py --data-dir mcap_data/fold_clothv2 --port 8765

然后在浏览器中打开 `http://localhost:8765`（远程开发环境需先做端口转发）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | 必填 | mcap 数据目录（含 `config.py`） |
| `--port` | `8765` | Web 服务端口 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--limit` | `0`（全部） | 最多扫描的 episode 数，`0` = 不限制 |

界面功能：
- 左侧列表按 folder 分组，支持关键词搜索；状态指示灯（绿/黄/红）标记 episode 质量
- 右侧播放选中 episode 的相机视频，进度条可拖拽跳帧
- **Delete 按钮**：直接从磁盘删除当前 episode（不可恢复，请谨慎使用）

键盘快捷键：

| 按键 | 功能 |
|---|---|
| `↑` / `↓` | 切换 episode |
| `Space` | 播放 / 暂停 |
| `←` / `→` | 快退 / 快进 2 秒 |
| `Delete` / `Backspace` | 删除当前 episode |

---

### 1.1.6 可视化数据与VF标签

离线可视化单个/多个 episode 的相机采样图、state/action 轨迹、`binned_value` 分布，以及（如果存在）`predicted_value`、`advantage`、`is_good_action`。

在 [`scripts/cmds/visualize.sh`](scripts/cmds/visualize.sh) 中修改参数后运行：

    bash scripts/cmds/visualize_values.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `./lerobot_data/Fold_clothes` | LeRobot 数据集目录 |
| `OUTPUT_DIR` | `./assets/visualizations` | 可视化图片输出目录 |
| `EPISODES` | `0,1,2` | 逗号分隔的 episode 编号；留空则处理全部 |
| `NUM_CAMERA_SAMPLES` | `5` | 每个 episode 相机采样帧数 |
| `SKIP_CAMERAS` | `false` | `true` = 跳过相机图像，加快速度 |


# 2. 预训练

## 2.1 奖励函数（Value Function）

### 2.1.1 训练奖励函数（单次训练）

在 [`scripts/cmds/vf_train.sh`](scripts/cmds/vf_train.sh) 中修改参数后运行：

    bash scripts/cmds/vf_train.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | 训练配置名 |
| `EXP_NAME` | `vf_v1` | 实验名（checkpoint 子目录） |
| `GPUS` | `0,1,2,3` | 使用的 GPU |
| `NUM_TRAIN_STEPS` | `20000` | 训练步数 |
| `OVERWRITE` | `true` | `false` = resume |

### 2.1.2 K-fold 奖励函数交叉验证训练

单个 VF 在全数据集训练后对全数据集打分会过拟合。K-fold 将数据集分成 K 份，训练 K 个 VF（每个在 K-1 份上训练），每个 VF 只在自己的留出集上打分。

**步骤 1：add_labels 时指定 NUM_FOLDS（见 §1.1.2）**

确保 `scripts/cmds/add_labels.sh` 中 `NUM_FOLDS=3` 已设置，运行后会生成 `meta/folds.json`。

**步骤 2：K-fold 训练（Phase 1）**

在 [`scripts/cmds/vf_kfold_train.sh`](scripts/cmds/vf_kfold_train.sh) 中修改参数后运行：

    bash scripts/cmds/vf_kfold_train.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `REPO_ID` | `Fold_clothes` | LeRobot 数据集 ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | VF 训练配置名 |
| `GPUS` | `(2 3 4 5 6 7)` | 可用 GPU 数组 |
| `NUM_FOLDS` | `3` | K 值 |
| `NUM_TRAIN_STEPS` | `20000` | 每个 VF 的训练步数 |
| `GPUS_PER_FOLD` | `2` | 每个 fold 使用的 GPU 数 |
| `RESUME` | `false` | `true` = 从已有 checkpoint 继续 |

**步骤 3：K-fold 推理 + 合并（Phase 2+3）**

在 [`scripts/cmds/vf_kfold_label.sh`](scripts/cmds/vf_kfold_label.sh) 中修改参数后运行：

    bash scripts/cmds/vf_kfold_label.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `REPO_ID` | `Fold_clothes` | LeRobot 数据集 ID |
| `VF_CONFIG` | `pi06_rl_vf_airbot_clothes_folding` | VF 配置名 |
| `GPUS` | `(2 3 4 5 6 7)` | 可用 GPU 数组 |
| `NUM_FOLDS` | `3` | K 值 |
| `GPUS_PER_FOLD` | `2` | 每个 fold 使用的 GPU 数 |
| `CHECKPOINT_STEP` | 自动检测最新 | 留空自动取最大数字子目录（如 19999） |
| `POSITIVE_FRACTION` | `0.3` | 正样本比例（预训练 0.3，微调 0.4） |
| `GAMMA` | `0.99` | advantage 折扣因子 |

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

| 参数（在 sh 中修改） | 默认值 | 说明 |
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

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `FUNC_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | 训练配置名 |

### 2.2.2 优势条件策略训练

使用 `is_good_action` 标签进行优势条件策略训练，训练时将 `Advantage: Positive/Negative` 注入 prompt，30% 概率 dropout（用于推理时 CFG）。

在 [`scripts/cmds/train_policy.sh`](scripts/cmds/train_policy.sh) 中修改参数后运行：

    bash scripts/cmds/train_policy.sh

| 参数（在 sh 中修改） | 默认值 | 说明 |
|---|---|---|
| `POLICY_CONFIG` | `pi06_rl_pretrain_airbot_clothes_folding` | 训练配置名 |
| `EXP_NAME` | `policy_iter0` | 实验名 |
| `GPUS` | `0,1,2,3,4,5,6,7` | 使用的 GPU |
| `OVERWRITE` | `true` | `false` = resume |

- 推理时使用 Classifier-Free Guidance: `ε_guided = ε_uncond + w × (ε_positive − ε_uncond)`，w > 1（论文推荐 w = 2）


# 3. 数据轮转

支持在已标注的数据集上增量追加新 mcap 数据，重新标注后继续训练，无需从头重建。
对应论文的迭代循环：收集新数据 → 追加 → 重标注 → K-fold 训练 VF + 重标 is_good_action → 重训 Policy。

VF 标注统一使用 K-fold 交叉验证（见 §2.1.2），避免单 VF 在训练集上打分过拟合。

### 3.1 Iteration 0：初始数据 → 完整 pipeline

    # 1) mcap → lerobot（修改 convert_mcap.sh 中的 DATA_DIR）
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

收集新 mcap 后，有两种方式追加数据（选其一）：

**方式 A：新 mcap 放入独立目录**（推荐，无需移动文件）
- 将新 mcap 保存在独立目录（如 `mcap_data/fold_clothv2_dagger1/`），该目录有自己的 `config.py`（`TASK_NAME` 须与已有数据集一致）
- 在 `convert_mcap.sh` 中：`DATA_DIR=mcap_data/fold_clothv2_dagger1`，`RESUME=true`，`SKIP_EPISODES=0`

**方式 B：新 mcap 合并到原目录**（旧方式）
- 将新 mcap 文件夹移入原 `mcap_data/fold_clothv2/`，更新其 `config.py`：
  1. 新 mcap 文件夹加入 `FOLDERS`
  2. 更新 `FAILED_EPISODES` 覆盖所有 episode（新旧皆需）
  3. 如有人类纠正，更新 `INTERVENTION_EPISODES`
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

**Step 2：运行机械臂推理**

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

## 4.2 DAgger 在线干预RL数据集采集

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

启用 `RECORD=true` 后，每个 episode 自动保存为 MCAP 文件，格式兼容 `convert_mcap_data_to_lerobot.py` 转换脚本。录制内容包括：

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

# 2) 更新 config.py 中的 INTERVENTION_EPISODES（DAgger intervention=1 的步会自动标记）
# 3) 继续 §3.2 的重标注 → 重训流程
```

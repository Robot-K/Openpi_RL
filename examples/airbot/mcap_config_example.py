# ============================================================
# mcap_data/config.py — Unified configuration for the full
# data-conversion → returns-annotation → VF-training pipeline.
# ============================================================

# ---------- 1. 数据采集 / mcap 转换参数 ----------
TASK_NAME = "Fold_clothes"
MCAP_TASK_NAME = "Fold_clothes"  # task name written in MCAP metahome/userfb800e0c/projects
ROBOT_TYPE = "airbot"
FOLDERS = [
    "fold_clothv2_0_10",
    "fold_clothv2_10_20",
    "fold_clothv2_20_30",
    "fold_clothv2_30_40",
    "fold_clothv2_40_50",
]

STATE_TOPICS = [
    "left/follow/arm/joint_state/position",
    "left/follow/eef/joint_state/position",
    "right/follow/arm/joint_state/position",
    "right/follow/eef/joint_state/position",
]
ACTION_TOPICS = [
    "left/lead/arm/joint_state/position",
    "left/lead/eef/joint_state/position",
    "right/lead/arm/joint_state/position",
    "right/lead/eef/joint_state/position",
]
CAMERA_TOPICS = {
    "base_0_rgb": "/env_camera/color/image_raw",
    "left_wrist_0_rgb": "/left_camera/color/image_raw",
    "right_wrist_0_rgb": "/right_camera/color/image_raw",
}
FPS = 20

# ---------- 2. 标注参数 (only used by add_labels) ----------
FAILED_EPISODES = []           # These episodes are marked failed (overrides SUCCESS_EPISODES)
INTERVENTION_EPISODES = {}     # {ep_idx: [[start, end], ...]} for partial interventions
STAGE_BOUNDARIES = (200,)      # Tuple of episode indices where stages change. E.g. (100,) means 0-99 is stage 1, 100+ is stage 2.

#!/usr/bin/env bash
# 打印所有 mcap->lerobot 转换任务的当前进度（单次）。
# 实时刷新： watch -n 5 bash scripts/cmds/watch_convert.sh
cd "$(dirname "$0")/../.."

# 任务名 -> 日志 glob（[0-9] 确保 wuxi 与 wuxi_airdc 不串台）
print_one() {
    local name="$1" glob="$2"
    local log; log=$(ls -t $glob 2>/dev/null | head -1)
    if [[ -z "$log" ]]; then printf "  %-26s (未启动)\n" "$name"; return; fi
    local line cur total pct last running
    line=$(grep -E '^\[[0-9]+/[0-9]+\] Converting' "$log" | tail -1)
    cur=$(sed -nE 's/^\[([0-9]+)\/[0-9]+\].*/\1/p' <<<"$line")
    total=$(sed -nE 's/^\[[0-9]+\/([0-9]+)\].*/\1/p' <<<"$line")
    last=$(sed -nE 's/^\[[0-9]+\/[0-9]+\] Converting (.*) \(.*/\1/p' <<<"$line")
    pct=""; [[ -n "$cur" && -n "$total" ]] && pct=$(( cur * 100 / total ))
    # 运行中判断：日志在最近 120s 内有写入 = 还在跑
    local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if (( age < 120 )); then running="▶ 运行中"; else running="■ ${age}s 无更新"; fi
    printf "  %-26s %5s/%-5s %3s%%  %-8s %s\n" "$name" "${cur:-0}" "${total:-?}" "${pct:-?}" "$running" "$last"
}

echo "===== mcap -> lerobot 转换进度  $(date '+%H:%M:%S') ====="
print_one "Fold_clothes(原始)"        "logs/convert_mcap_[0-9]*.log"
print_one "fold_clothv3_dagger"        "logs/convert_fold_clothv3_dagger_[0-9]*.log"
print_one "fold_clothv3_wam_dagger"    "logs/convert_fold_clothv3_wam_dagger_[0-9]*.log"
print_one "fold_clothv3_wuxi"          "logs/convert_fold_clothv3_wuxi_[0-9]*.log"
print_one "fold_clothv3_wuxi_airdc"    "logs/convert_fold_clothv3_wuxi_airdc_[0-9]*.log"
print_one "fold_clothv3_wuxi_dagger"   "logs/convert_fold_clothv3_wuxi_dagger_[0-9]*.log"
echo "磁盘: $(df -h /data | awk 'NR==2{print $4" 可用 ("$5" 已用)"}')"

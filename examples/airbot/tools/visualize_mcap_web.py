"""Web-based MCAP data visualizer for inspecting and cleaning robot episodes.

Usage:
    python scripts/tools/visualize_mcap_web.py --data_dir mcap_data/fold_clothes --port 8765

Then open http://localhost:8765 in Cursor's Simple Browser (or any browser with port forwarding).
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from mcap.reader import make_reader

# ---------------------------------------------------------------------------
# MCAP helpers (reused from convert_mcap_data_to_lerobot.py)
# ---------------------------------------------------------------------------

def decode_float_array(data: bytes) -> np.ndarray:
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    tab = flatbuffers.table.Table(bytearray(data), root_offset)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        return tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return np.array([], dtype=np.float32)


@dataclass
class TaskConfig:
    task_name: str
    robot_type: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]
    fps: int


def load_task_config(config_path: str | Path) -> TaskConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
    spec = importlib.util.spec_from_file_location("_mcap_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mcap_cfg"] = mod
    spec.loader.exec_module(mod)
    return TaskConfig(
        task_name=mod.TASK_NAME,
        robot_type=mod.ROBOT_TYPE,
        folders=mod.FOLDERS,
        state_topics=mod.STATE_TOPICS,
        action_topics=mod.ACTION_TOPICS,
        camera_topics=mod.CAMERA_TOPICS,
        fps=mod.FPS,
    )


# ---------------------------------------------------------------------------
# Scanning logic
# ---------------------------------------------------------------------------

@dataclass
class EpisodeInfo:
    folder: str
    filename: str
    path: str
    num_frames: int
    num_video_frames: dict[str, int]
    num_cameras_found: int
    expected_cameras: int
    has_metadata: bool
    error: str | None = None


def count_video_frames(video_bytes: bytes) -> int:
    """Write bytes to a temp file and count frames with OpenCV."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    finally:
        Path(tmp).unlink(missing_ok=True)


def scan_mcap_file(mcap_path: Path, config: TaskConfig) -> EpisodeInfo:
    """Quickly scan a single MCAP file for metadata without decoding all frames."""
    folder = mcap_path.parent.name
    filename = mcap_path.name
    info = EpisodeInfo(
        folder=folder,
        filename=filename,
        path=str(mcap_path),
        num_frames=0,
        num_video_frames={},
        num_cameras_found=0,
        expected_cameras=len(config.camera_topics),
        has_metadata=False,
    )
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)

            for attach in reader.iter_attachments():
                if attach.media_type == "video/mp4" and attach.name in config.camera_topics.values():
                    info.num_video_frames[attach.name] = count_video_frames(attach.data)
            info.num_cameras_found = len(info.num_video_frames)

            for md in reader.iter_metadata():
                if md.name == "task_info":
                    info.has_metadata = True

            msg_count = 0
            for _schema, _channel, _msg in reader.iter_messages(
                topics=config.state_topics + config.action_topics
            ):
                msg_count += 1
            num_topics = len(config.state_topics) + len(config.action_topics)
            info.num_frames = msg_count // num_topics if num_topics else 0

    except Exception as e:
        info.error = str(e)
    return info


def scan_all(data_dir: Path, config: TaskConfig, limit: int = 0) -> list[EpisodeInfo]:
    episodes = []
    for folder_name in sorted(config.folders):
        folder_path = data_dir / folder_name
        if not folder_path.is_dir():
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".mcap"):
                continue
            if limit > 0 and len(episodes) >= limit:
                break
            mcap_path = folder_path / fname
            print(f"  Scanning {folder_name}/{fname} ...", end="", flush=True)
            info = scan_mcap_file(mcap_path, config)
            tag = f" frames={info.num_frames}"
            if info.error:
                tag += f" ERROR={info.error}"
            print(tag)
            episodes.append(info)
        if limit > 0 and len(episodes) >= limit:
            break
    return episodes


# ---------------------------------------------------------------------------
# Frame extraction for viewing
# ---------------------------------------------------------------------------

def extract_base_frame_jpeg(mcap_path: Path, frame_idx: int, camera_topic: str) -> bytes | None:
    """Extract a single frame from the base camera as JPEG bytes."""
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for attach in reader.iter_attachments():
                if attach.media_type == "video/mp4" and attach.name == camera_topic:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(attach.data)
                        tmp_path = tmp.name
                    try:
                        cap = cv2.VideoCapture(tmp_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            return buf.tobytes()
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass
    return None


def extract_base_video_mp4(mcap_path: Path, camera_topic: str) -> bytes | None:
    """Extract the raw MP4 bytes for the base camera from the MCAP file."""
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for attach in reader.iter_attachments():
                if attach.media_type == "video/mp4" and attach.name == camera_topic:
                    return attach.data
    except Exception:
        pass
    return None


def extract_state_action(mcap_path: Path, config: TaskConfig) -> dict:
    """Extract state and action arrays for the whole episode."""
    states, actions = [], []
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            cnt_topics = {t: 0 for t in config.state_topics + config.action_topics}
            state_msg, action_msg = {}, {}
            cnt = 0

            for _schema, channel, message in reader.iter_messages(
                topics=config.state_topics + config.action_topics
            ):
                cnt_topics[channel.topic] += 1
                if cnt_topics[channel.topic] - cnt == 2:
                    s = np.concatenate([state_msg[t] for t in config.state_topics])
                    a = np.concatenate([action_msg[t] for t in config.action_topics])
                    states.append(s)
                    actions.append(a)
                    cnt += 1
                if channel.topic in config.state_topics:
                    state_msg[channel.topic] = decode_float_array(message.data)
                if channel.topic in config.action_topics:
                    action_msg[channel.topic] = decode_float_array(message.data)

            if state_msg and action_msg:
                s = np.concatenate([state_msg[t] for t in config.state_topics])
                a = np.concatenate([action_msg[t] for t in config.action_topics])
                states.append(s)
                actions.append(a)
    except Exception:
        pass

    return {
        "states": np.array(states).tolist() if states else [],
        "actions": np.array(actions).tolist() if actions else [],
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCAP Data Viewer</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
          --text2: #8b949e; --accent: #58a6ff; --red: #f85149; --green: #3fb950;
          --yellow: #d29922; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }

  .layout { display: flex; height: 100vh; }

  /* ---- Left panel ---- */
  .left-panel { width: 320px; min-width: 260px; border-right: 1px solid var(--border);
                display: flex; flex-direction: column; background: var(--card); }
  .left-header { padding: 12px 16px; border-bottom: 1px solid var(--border); }
  .left-header h1 { font-size: 1rem; margin-bottom: 4px; }
  .left-header .subtitle { font-size: 0.75rem; color: var(--text2); }
  .left-header input { width: 100%; margin-top: 8px; padding: 6px 10px; font-size: 0.8rem;
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); }
  .left-header input:focus { outline: none; border-color: var(--accent); }
  .ep-list { flex: 1; overflow-y: auto; }
  .folder-group { }
  .folder-header { padding: 8px 16px; font-size: 0.75rem; font-weight: 600; color: var(--accent);
                   background: rgba(88,166,255,0.05); border-bottom: 1px solid var(--border);
                   position: sticky; top: 0; z-index: 1; cursor: pointer; user-select: none;
                   display: flex; justify-content: space-between; align-items: center; }
  .folder-header .folder-count { color: var(--text2); font-weight: normal; }
  .ep-item { padding: 8px 16px 8px 28px; cursor: pointer; border-bottom: 1px solid var(--border);
             transition: background 0.1s; display: flex; justify-content: space-between; align-items: center; }
  .ep-item:hover { background: rgba(88,166,255,0.06); }
  .ep-item.active { background: rgba(88,166,255,0.12); border-left: 3px solid var(--accent); }
  .ep-item .ep-name { font-size: 0.82rem; font-weight: 500; white-space: nowrap;
                      overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
  .ep-item .ep-frames { font-size: 0.72rem; color: var(--text2); white-space: nowrap; }
  .ep-item .ep-status { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-left: 8px; }
  .ep-item .ep-status.ok { background: var(--green); }
  .ep-item .ep-status.warn { background: var(--yellow); }
  .ep-item .ep-status.err { background: var(--red); }

  /* ---- Right panel ---- */
  .right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .right-header { padding: 12px 20px; border-bottom: 1px solid var(--border);
                  display: flex; justify-content: space-between; align-items: center; min-height: 52px; }
  .right-header .title { font-size: 0.95rem; font-weight: 600; }
  .right-header .info { font-size: 0.8rem; color: var(--text2); }
  .btn { padding: 6px 16px; border-radius: 6px; border: 1px solid var(--border); cursor: pointer;
         font-size: 0.82rem; background: var(--card); color: var(--text); transition: all 0.15s; }
  .btn:hover { border-color: var(--accent); }
  .btn-danger { border-color: var(--red); color: var(--red); }
  .btn-danger:hover { background: var(--red); color: #fff; }

  .video-area { flex: 1; display: flex; flex-direction: column; align-items: center;
                justify-content: center; padding: 20px; overflow: hidden; }
  .video-area video { max-width: 100%; max-height: calc(100vh - 180px); border-radius: 8px;
                      background: #000; display: block; }
  .video-area .placeholder { color: var(--text2); font-size: 0.9rem; }

  .controls { padding: 12px 20px; border-top: 1px solid var(--border);
              display: flex; align-items: center; gap: 12px; }
  .controls input[type=range] { flex: 1; accent-color: var(--accent); cursor: pointer; height: 6px; }
  .controls .frame-info { font-size: 0.8rem; color: var(--text2); min-width: 100px; text-align: right; }

  .keyboard-hint { padding: 6px 20px; border-top: 1px solid var(--border); font-size: 0.7rem;
                   color: var(--text2); text-align: center; background: var(--card); }
</style>
</head>
<body>

<div class="layout">
  <!-- Left: episode list -->
  <div class="left-panel">
    <div class="left-header">
      <h1 id="taskName">MCAP Viewer</h1>
      <div class="subtitle" id="subtitle"></div>
      <input type="text" id="searchInput" placeholder="Search...">
    </div>
    <div class="ep-list" id="epList"></div>
  </div>

  <!-- Right: video viewer -->
  <div class="right-panel">
    <div class="right-header">
      <div>
        <span class="title" id="epTitle">Select an episode</span>
        <span class="info" id="epInfo"></span>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <button class="btn btn-danger" id="deleteBtn" style="display:none">Delete</button>
      </div>
    </div>
    <div class="video-area">
      <video id="player" controls muted loop></video>
      <div class="placeholder" id="placeholder">Click an episode on the left to start viewing</div>
    </div>
    <div class="controls" id="controlBar" style="display:none">
      <input type="range" id="scrubber" min="0" max="100" value="0" step="0.1">
      <div class="frame-info" id="frameInfo">0:00 / 0:00</div>
    </div>
    <div class="keyboard-hint">Arrow Up/Down: switch episode &nbsp;|&nbsp; Space: play/pause &nbsp;|&nbsp; Arrow Left/Right: seek +/-2s &nbsp;|&nbsp; Delete/Backspace: delete current</div>
  </div>
</div>

<script>
const API = '';
let episodes = [];
let currentIdx = -1;

async function loadEpisodes() {
  const resp = await fetch(API + '/api/episodes');
  const data = await resp.json();
  episodes = data.episodes;
  document.getElementById('taskName').textContent = data.task_name;
  document.getElementById('subtitle').textContent = data.total + ' episodes';
  renderList();
  if (episodes.length > 0 && currentIdx < 0) selectEpisode(0);
}

function getStatus(ep) {
  if (ep.error) return 'err';
  if (ep.num_cameras_found < ep.expected_cameras || ep.num_frames < 10) return 'warn';
  return 'ok';
}

function renderList() {
  const search = document.getElementById('searchInput').value.toLowerCase();
  const container = document.getElementById('epList');
  const groups = {};
  episodes.forEach((ep, i) => {
    if (search && !ep.filename.toLowerCase().includes(search) && !ep.folder.toLowerCase().includes(search)) return;
    if (!groups[ep.folder]) groups[ep.folder] = [];
    groups[ep.folder].push({ep, i});
  });
  let html = '';
  for (const [folder, items] of Object.entries(groups)) {
    html += `<div class="folder-group">
      <div class="folder-header">${folder} <span class="folder-count">${items.length}</span></div>`;
    for (const {ep, i} of items) {
      const st = getStatus(ep);
      html += `<div class="ep-item ${i === currentIdx ? 'active' : ''}" data-idx="${i}">
        <div>
          <div class="ep-name" title="${ep.folder}/${ep.filename}">${ep.filename}</div>
          <div class="ep-frames">${ep.num_frames} frames</div>
        </div>
        <div class="ep-status ${st}"></div>
      </div>`;
    }
    html += '</div>';
  }
  container.innerHTML = html;
}

document.getElementById('searchInput').addEventListener('input', renderList);

document.getElementById('epList').addEventListener('click', function(e) {
  const item = e.target.closest('.ep-item');
  if (!item) return;
  selectEpisode(parseInt(item.dataset.idx));
});

function selectEpisode(idx) {
  if (idx < 0 || idx >= episodes.length) return;
  currentIdx = idx;
  const ep = episodes[idx];
  const name = ep.folder + '/' + ep.filename;

  document.getElementById('epTitle').textContent = name;
  document.getElementById('epInfo').textContent = '  ' + ep.num_frames + ' frames | cam ' + ep.num_cameras_found + '/' + ep.expected_cameras;
  document.getElementById('deleteBtn').style.display = 'inline-block';
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('controlBar').style.display = 'flex';

  const player = document.getElementById('player');
  player.style.display = 'block';
  player.src = API + '/api/video?path=' + encodeURIComponent(ep.path);
  player.play().catch(() => {});

  renderList();

  const active = document.querySelector('.ep-item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

// Video scrubber sync
const player = document.getElementById('player');
const scrubber = document.getElementById('scrubber');

player.addEventListener('loadedmetadata', () => {
  scrubber.max = player.duration || 100;
  scrubber.value = 0;
  updateFrameInfo();
});
player.addEventListener('timeupdate', () => {
  if (!scrubbing) {
    scrubber.value = player.currentTime;
  }
  updateFrameInfo();
});

let scrubbing = false;
let wasPlaying = false;
scrubber.addEventListener('pointerdown', () => {
  scrubbing = true;
  wasPlaying = !player.paused;
  player.pause();
});
scrubber.addEventListener('input', () => {
  player.currentTime = parseFloat(scrubber.value);
  updateFrameInfo();
});
scrubber.addEventListener('pointerup', () => {
  scrubbing = false;
  player.currentTime = parseFloat(scrubber.value);
  if (wasPlaying) player.play();
});
scrubber.addEventListener('change', () => {
  scrubbing = false;
  player.currentTime = parseFloat(scrubber.value);
  if (wasPlaying) player.play();
});

function updateFrameInfo() {
  const cur = player.currentTime || 0;
  const dur = player.duration || 0;
  document.getElementById('frameInfo').textContent = fmtTime(cur) + ' / ' + fmtTime(dur);
}
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

// Delete
document.getElementById('deleteBtn').addEventListener('click', async () => {
  if (currentIdx < 0) return;
  const ep = episodes[currentIdx];
  if (!confirm('Delete ' + ep.folder + '/' + ep.filename + '?')) return;
  const resp = await fetch(API + '/api/delete', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({paths: [ep.path]})
  });
  const result = await resp.json();
  if (result.deleted.length) {
    const nextIdx = Math.min(currentIdx, episodes.length - 2);
    currentIdx = -1;
    await loadEpisodes();
    if (episodes.length > 0) selectEpisode(Math.max(0, nextIdx));
  } else {
    alert('Failed: ' + JSON.stringify(result.failed));
  }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowDown') { e.preventDefault(); selectEpisode(currentIdx + 1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); selectEpisode(currentIdx - 1); }
  else if (e.key === ' ') { e.preventDefault(); player.paused ? player.play() : player.pause(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); player.currentTime = Math.max(0, player.currentTime - 2); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); player.currentTime = Math.min(player.duration, player.currentTime + 2); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); document.getElementById('deleteBtn').click(); }
});

loadEpisodes();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

app = Flask(__name__)
DATA_DIR: Path = Path(".")
CONFIG: TaskConfig | None = None
EPISODES: list[EpisodeInfo] = []
BASE_CAMERA_TOPIC: str = ""
VIDEO_CACHE: dict[str, bytes] = {}  # path -> compressed video bytes


def compress_video(video_bytes: bytes, height: int = 480, crf: int = 28) -> bytes:
    """Re-encode video at lower resolution/quality via ffmpeg for faster preview."""
    cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-an",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, input=video_bytes, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            orig_kb = len(video_bytes) // 1024
            comp_kb = len(result.stdout) // 1024
            print(f"  compressed {orig_kb} KB -> {comp_kb} KB")
            return result.stdout
    except Exception as e:
        print(f"  ffmpeg compression failed: {e}, serving original")
    return video_bytes


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/episodes")
def api_episodes():
    return jsonify({
        "task_name": CONFIG.task_name,
        "total": len(EPISODES),
        "episodes": [
            {
                "folder": ep.folder,
                "filename": ep.filename,
                "path": ep.path,
                "num_frames": ep.num_frames,
                "num_video_frames": ep.num_video_frames,
                "num_cameras_found": ep.num_cameras_found,
                "expected_cameras": ep.expected_cameras,
                "has_metadata": ep.has_metadata,
                "error": ep.error,
            }
            for ep in EPISODES
        ],
    })


@app.route("/api/video")
def api_video():
    path = request.args.get("path", "")
    if not path:
        return "Missing path", 400
    mcap_path = Path(path)
    if not mcap_path.is_file():
        return "File not found", 404

    if path not in VIDEO_CACHE:
        raw = extract_base_video_mp4(mcap_path, BASE_CAMERA_TOPIC)
        if raw is None:
            return "Could not extract video", 500
        VIDEO_CACHE[path] = compress_video(raw)
    video_bytes = VIDEO_CACHE[path]

    total = len(video_bytes)
    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte_start = int(m.group(1))
            byte_end = int(m.group(2)) if m.group(2) else total - 1
            byte_end = min(byte_end, total - 1)
            chunk = video_bytes[byte_start:byte_end + 1]
            return Response(
                chunk,
                status=206,
                mimetype="video/mp4",
                headers={
                    "Content-Range": f"bytes {byte_start}-{byte_end}/{total}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(chunk)),
                },
            )

    return Response(
        video_bytes,
        mimetype="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total),
        },
    )


@app.route("/api/frame")
def api_frame():
    path = request.args.get("path", "")
    frame_idx = int(request.args.get("frame", 0))
    if not path:
        return "Missing path", 400
    mcap_path = Path(path)
    if not mcap_path.is_file():
        return "File not found", 404

    jpg = extract_base_frame_jpeg(mcap_path, frame_idx, BASE_CAMERA_TOPIC)
    if jpg is None:
        return "Could not extract frame", 500
    return Response(jpg, mimetype="image/jpeg")


@app.route("/api/state_action")
def api_state_action():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"states": [], "actions": []})
    mcap_path = Path(path)
    if not mcap_path.is_file():
        return jsonify({"states": [], "actions": []})
    return jsonify(extract_state_action(mcap_path, CONFIG))


@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.get_json()
    paths = data.get("paths", [])
    deleted, failed = [], []
    for p in paths:
        try:
            Path(p).unlink()
            deleted.append(p)
        except Exception as e:
            failed.append({"path": p, "error": str(e)})

    global EPISODES
    deleted_set = set(deleted)
    EPISODES = [ep for ep in EPISODES if ep.path not in deleted_set]
    for p in deleted_set:
        VIDEO_CACHE.pop(p, None)

    return jsonify({"deleted": deleted, "failed": failed})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MCAP Data Web Viewer")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to mcap data directory (e.g. mcap_data/fold_clothes)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to scan (0 = all)")
    args = parser.parse_args()

    global DATA_DIR, CONFIG, EPISODES, BASE_CAMERA_TOPIC
    DATA_DIR = Path(args.data_dir)
    CONFIG = load_task_config(DATA_DIR / "config.py")
    BASE_CAMERA_TOPIC = CONFIG.camera_topics.get("base_0_rgb", list(CONFIG.camera_topics.values())[0])

    print(f"Task: {CONFIG.task_name}")
    print(f"Base camera topic: {BASE_CAMERA_TOPIC}")
    limit_msg = f" (limit: {args.limit})" if args.limit > 0 else ""
    print(f"Scanning {len(CONFIG.folders)} folders...{limit_msg}")
    EPISODES = scan_all(DATA_DIR, CONFIG, limit=args.limit)
    print(f"\nDone. {len(EPISODES)} episodes loaded.")
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"If using Cursor remote, forward port {args.port} and open in Simple Browser.\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

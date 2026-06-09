"""Fix swapped camera attachments in MCAP files.

In the specified folder range, the ``/env_camera/color/image_raw`` and
``/left_camera/color/image_raw`` video attachments are swapped.  This script
rewrites each affected ``.mcap`` file with the two attachment names corrected.

The original file is backed up to ``<name>.mcap.bak`` before overwriting.

Usage:
    python scripts/tools/fix_swapped_cameras.py \
        --dat-dir mcap_data/fold_clothv2 \
        --start-folder fold_clothv2_670_680 \
        --end-folder fold_clothv2_700_710

    # Dry run (only report, don't modify):
    python scripts/tools/fix_swapped_cameras.py \
        --dat-dir mcap_data/fold_clothv2 \
        --start-folder fold_clothv2_670_680 \
        --end-folder fold_clothv2_700_710 \
        --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import shutil
import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer

_DA = "\x64\x61\x74\x61"  # attribute name used by mcap objects
_MD = "\x6d\x65\x74\x61" + _DA  # channel kwarg for extra info


def _get(obj, fallback=None):
    """Read the binary payload attribute from an mcap record."""
    return getattr(obj, _DA, fallback)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_folders(config_path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("_mcap_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mcap_cfg"] = mod
    spec.loader.exec_module(mod)
    return mod.FOLDERS


# ---------------------------------------------------------------------------
# Core: rewrite one MCAP with swapped attachment names
# ---------------------------------------------------------------------------

SWAP_A = "/env_camera/color/image_raw"
SWAP_B = "/left_camera/color/image_raw"


def rewrite_mcap_swap_cameras(src: Path, dst: Path) -> None:
    """Read *src*, swap the two camera attachment names, write to *dst*."""

    with src.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        # --- collect schemas (preserve original order) ---
        schemas = {}  # old_id -> (name, encoding, payload)
        for sid, sc in sorted(summary.schemas.items()):
            schemas[sid] = (sc.name, sc.encoding, _get(sc))

        # --- collect channels ---
        channels = {}  # old_id -> (topic, msg_encoding, old_schema_id, meta)
        for cid, ch in sorted(summary.channels.items()):
            channels[cid] = (ch.topic, ch.message_encoding, ch.schema_id,
                             getattr(ch, _MD))

        # --- collect messages ---
        f.seek(0)
        reader2 = make_reader(f)
        messages = []
        for _schema, channel, msg in reader2.iter_messages():
            messages.append((channel.id, msg.log_time, msg.publish_time,
                             msg.sequence, _get(msg)))

        # --- collect attachments ---
        f.seek(0)
        reader3 = make_reader(f)
        attachments = []
        for attach in reader3.iter_attachments():
            attachments.append((attach.name, attach.media_type,
                                attach.create_time, attach.log_time,
                                _get(attach)))

    # --- swap attachment names ---
    swapped = []
    for name, media_type, create_time, log_time, payload in attachments:
        if name == SWAP_A:
            name = SWAP_B
        elif name == SWAP_B:
            name = SWAP_A
        swapped.append((name, media_type, create_time, log_time, payload))

    # --- write new mcap ---
    buf = io.BytesIO()
    writer = Writer(buf)
    writer.start()

    # register schemas, mapping old_id -> new_id
    old_to_new_schema: dict[int, int] = {}
    for old_sid, (name, encoding, payload) in schemas.items():
        new_sid = writer.register_schema(name=name, encoding=encoding,
                                         **{_DA: payload})
        old_to_new_schema[old_sid] = new_sid

    # register channels, mapping old_id -> new_id
    old_to_new_channel: dict[int, int] = {}
    for old_cid, (topic, msg_enc, old_sid, meta) in channels.items():
        new_cid = writer.register_channel(
            topic=topic,
            message_encoding=msg_enc,
            schema_id=old_to_new_schema[old_sid],
            **{_MD: meta},
        )
        old_to_new_channel[old_cid] = new_cid

    # write messages
    for old_cid, log_time, publish_time, sequence, payload in messages:
        writer.add_message(
            channel_id=old_to_new_channel[old_cid],
            log_time=log_time,
            publish_time=publish_time,
            sequence=sequence,
            **{_DA: payload},
        )

    # write attachments (with swapped names)
    for name, media_type, create_time, log_time, payload in swapped:
        writer.add_attachment(
            create_time=create_time,
            log_time=log_time,
            name=name,
            media_type=media_type,
            **{_DA: payload},
        )

    writer.finish()

    dst.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fix swapped env/left camera attachments in MCAP files")
    parser.add_argument("--dat-dir", type=str, required=True,
                        help="Root directory containing config.py and folders")
    parser.add_argument("--start-folder", type=str, required=True,
                        help="First folder to fix (inclusive)")
    parser.add_argument("--end-folder", type=str, required=True,
                        help="Last folder to fix (inclusive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report files that would be fixed")
    args = parser.parse_args()

    dat_dir = Path(args.dat_dir)
    config_path = dat_dir / "config.py"
    folders = load_folders(config_path)

    # Determine range
    try:
        start_idx = folders.index(args.start_folder)
    except ValueError:
        print(f"ERROR: start folder '{args.start_folder}' not found in config")
        sys.exit(1)
    try:
        end_idx = folders.index(args.end_folder)
    except ValueError:
        print(f"ERROR: end folder '{args.end_folder}' not found in config")
        sys.exit(1)

    target_folders = folders[start_idx : end_idx + 1]
    print(f"Folders to fix: {len(target_folders)}"
          f"  ({target_folders[0]} .. {target_folders[-1]})")
    print(f"Swapping: {SWAP_A} <-> {SWAP_B}")
    if args.dry_run:
        print("[DRY RUN]")
    print()

    total = 0
    for folder_name in target_folders:
        folder_path = dat_dir / folder_name
        if not folder_path.is_dir():
            print(f"  SKIP (not a directory): {folder_name}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".mcap"):
                continue
            mcap_path = folder_path / fname
            short = f"{folder_name}/{fname}"

            if args.dry_run:
                print(f"  [DRY] would fix {short}")
            else:
                bak_path = mcap_path.with_suffix(".mcap.bak")
                tmp_path = mcap_path.with_suffix(".mcap.tmp")
                print(f"  Fixing {short} ...", end="", flush=True)
                try:
                    rewrite_mcap_swap_cameras(mcap_path, tmp_path)
                    shutil.copy2(mcap_path, bak_path)
                    tmp_path.replace(mcap_path)
                    print(" OK")
                except Exception as e:
                    print(f" ERROR: {e}")
                    tmp_path.unlink(missing_ok=True)
            total += 1

    print(f"\nDone. {total} file(s) {'would be ' if args.dry_run else ''}processed.")


if __name__ == "__main__":
    main()

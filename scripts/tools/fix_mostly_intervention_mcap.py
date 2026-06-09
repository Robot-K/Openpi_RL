"""Fix raw MCAP episodes whose DAgger intervention is mostly 1s.

This is the MCAP-level counterpart of ``fix_all_intervention_episodes.py`` (which
operates on the converted LeRobot parquet). It rewrites the *source* mcap files in
place so that any later re-conversion already has the corrected labels.

For every ``*.mcap`` under --data_dir, read the values published on the
``/dagger/intervention`` topic. If the fraction of intervention==1 exceeds
--threshold (default 0.8), rewrite the file with every intervention message set
to 0 (all other records -- schemas, channels, video attachments, metadata,
state/action messages -- are copied verbatim).

Usage:
    # Dry-run: only report which episodes would be fixed.
    python scripts/tools/fix_all_intervention_mcap.py \\
        --data_dir mcap_data/fold_clothes --dry-run

    # Apply the fix in place (originals backed up to <file>.mcap.bak unless --no-backup).
    python scripts/tools/fix_all_intervention_mcap.py \\
        --data_dir mcap_data/fold_clothes --threshold 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
import numpy as np
from mcap.exceptions import EndOfFile
from mcap.exceptions import RecordLengthLimitExceeded
from mcap.reader import NonSeekingReader
from mcap.reader import make_reader


DAGGER_INTERVENTION_TOPIC = "/dagger/intervention"


# ---------------------------------------------------------------------------
# FlatBuffers airbot_fbs.FloatArray codec (matches convert_mcap_data_to_lerobot.py)
# ---------------------------------------------------------------------------

def decode_float_array(data: bytes) -> np.ndarray:
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    tab = flatbuffers.table.Table(bytearray(data), root_offset)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        return tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return np.array([], dtype=np.float32)


def encode_float_array(values: list[float]) -> bytes:
    """Encode a list of floats as an ``airbot_fbs.FloatArray`` FlatBuffers message."""
    b = flatbuffers.Builder(0)
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependFloat32(float(v))
    vec = b.EndVector()
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, vec, 0)
    obj = b.EndObject()
    b.Finish(obj)
    return bytes(b.Output())


# ---------------------------------------------------------------------------
# Safe iterators (fall back to a sequential reader on malformed summary/footer)
# ---------------------------------------------------------------------------

def iter_messages_safe(mcap_path: Path, topics: list[str] | None = None):
    try:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            yield from reader.iter_messages(topics=topics)
            return
    except RecordLengthLimitExceeded:
        pass
    try:
        with mcap_path.open("rb") as f:
            reader = NonSeekingReader(f)
            yield from reader.iter_messages(topics=topics)
    except EndOfFile:
        return


# ---------------------------------------------------------------------------
# Scan / rewrite
# ---------------------------------------------------------------------------

def intervention_fraction(mcap_path: Path) -> tuple[float, int]:
    """Return (fraction_of_ones, num_intervention_msgs) for one mcap file."""
    vals: list[int] = []
    for _schema, channel, message in iter_messages_safe(mcap_path, [DAGGER_INTERVENTION_TOPIC]):
        if channel.topic != DAGGER_INTERVENTION_TOPIC:
            continue
        decoded = decode_float_array(message.data)
        vals.append(int(round(decoded[0])) if len(decoded) > 0 else 0)
    if not vals:
        return 0.0, 0
    arr = np.asarray(vals, dtype=np.int64)
    return float((arr == 1).mean()), arr.size


def rewrite_intervention_zero(mcap_path: Path, backup: bool) -> None:
    """Copy *mcap_path* to a staging file, zeroing every intervention message,
    then atomically replace the original (optionally keeping a .bak copy).

    All non-intervention records (schemas, channels, video attachments, metadata,
    state/action messages) are copied verbatim and in log-time order.
    """
    from mcap.writer import Writer

    staging = mcap_path.with_suffix(mcap_path.suffix + ".staging")

    with mcap_path.open("rb") as f_in, staging.open("wb") as f_out:
        reader = make_reader(f_in)
        header = reader.get_header()
        writer = Writer(f_out)
        writer.start(profile=header.profile, library=header.library)

        # Map source schema/channel ids -> ids registered in the new writer.
        schema_map: dict[int, int] = {}
        channel_map: dict[int, int] = {}

        for schema, channel, message in reader.iter_messages():
            if schema is not None and schema.id not in schema_map:
                schema_map[schema.id] = writer.register_schema(
                    name=schema.name, encoding=schema.encoding, data=schema.data
                )
            if channel.id not in channel_map:
                new_schema_id = schema_map.get(schema.id, 0) if schema is not None else 0
                channel_map[channel.id] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    schema_id=new_schema_id,
                    metadata=channel.metadata,
                )

            data = message.data
            if channel.topic == DAGGER_INTERVENTION_TOPIC:
                # Preserve the original vector length, but force every value to 0.
                n = max(1, decode_float_array(data).size)
                data = encode_float_array([0.0] * n)

            writer.add_message(
                channel_id=channel_map[channel.id],
                log_time=message.log_time,
                data=data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )

        # Copy attachments (e.g. mp4 camera videos) and metadata unchanged.
        for attach in reader.iter_attachments():
            writer.add_attachment(
                create_time=attach.create_time,
                log_time=attach.log_time,
                name=attach.name,
                media_type=attach.media_type,
                data=attach.data,
            )
        for md in reader.iter_metadata():
            writer.add_metadata(name=md.name, data=md.metadata)

        writer.finish()

    if backup:
        backup_path = mcap_path.with_suffix(mcap_path.suffix + ".bak")
        if not backup_path.exists():
            mcap_path.replace(backup_path)
        else:
            print(f"  (backup {backup_path.name} already exists, not overwriting)")
    staging.replace(mcap_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=str,
                   help="Directory containing the mcap folders (scanned recursively for *.mcap).")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Fraction of intervention==1 above which the mcap is fixed (default 0.8).")
    p.add_argument("--no-backup", action="store_true",
                   help="Do not keep a <file>.mcap.bak copy of the original before overwriting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print which mcap files would be modified; do not write.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"data_dir not found: {data_dir}")

    paths = sorted(data_dir.rglob("*.mcap"))
    print(f"Scanning {len(paths)} mcap files in {data_dir}")

    to_fix: list[tuple[Path, float]] = []
    for i, path in enumerate(paths):
        try:
            frac, n = intervention_fraction(path)
        except Exception as e:
            print(f"  [{i}] ERROR reading {path}: {e}")
            continue
        if n == 0:
            continue
        if frac > args.threshold:
            to_fix.append((path, frac))
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(paths)} (found {len(to_fix)} mostly-1 episodes so far)")

    print(f"\nFound {len(to_fix)} mcap files with >{args.threshold:.0%} intervention.")
    for path, frac in to_fix[:30]:
        print(f"  {frac:6.1%}  {path.relative_to(data_dir)}")
    if len(to_fix) > 30:
        print(f"  ... (+{len(to_fix) - 30} more)")

    if args.dry_run:
        print("\n[dry-run] Skipping writes.")
        return
    if not to_fix:
        return

    print("\nRewriting mcap files...")
    for n, (path, _frac) in enumerate(to_fix, 1):
        try:
            rewrite_intervention_zero(path, backup=not args.no_backup)
        except Exception as e:
            print(f"  ERROR rewriting {path}: {e}")
            continue
        if n % 20 == 0 or n == len(to_fix):
            print(f"  rewrote {n}/{len(to_fix)}")

    print(f"\nDone. Fixed {len(to_fix)} mcap files.")


if __name__ == "__main__":
    main()

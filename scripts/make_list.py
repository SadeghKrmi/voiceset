#!/usr/bin/env python3
"""Stage 4: verified clips -> molana's training lists.

    python scripts/make_list.py [--val 0.01]

Writes <output>/molana/:
  train_list_en.txt, val_list_en.txt   <wav>|<transcript>|<speaker id>, molana's format
  wavs/                                every listed clip, in one folder as molana's
                                       root_path expects (hard links where possible)
  meta.json                            what built it: the vaguye that wrote the
                                       transcripts, clip counts and hours, and how
                                       often each symbol new to molana occurs

The transcript is the phonemes Kokoro spoke, in molana's conventions
(common.molana_transcript). Those conventions are vaguye's, so meta.json records
which vaguye it was: if vaguye changes them, rebuild the lists with it — the
audio does not change, only the transcripts.

Only clips whose sentence is still in the sentence files are listed, so deleting
a sentence takes it out of the dataset without touching any audio.
"""

import collections
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from common import (NEW_SYMBOLS, TRANSCRIPT_SYMBOLS, Manifest, clips, converted_wav,
                    load_config, molana_transcript, parse_args, reject)


def extra(parser):
    parser.add_argument("--val", type=float, default=0.01,
                        help="share of clips held out for validation (default: 0.01)")


def vaguye_source() -> dict:
    """Which vaguye is installed: its version, and the commit when it came from git."""
    dist = importlib.metadata.distribution("vaguye")
    source = {"version": dist.version}
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    source["url"] = direct.get("url")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if not commit and (source["url"] or "").startswith("file://"):
        folder = source["url"][len("file://"):]
        git = subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", folder, "status", "--porcelain", "src"],
                               capture_output=True, text=True).stdout.strip()
        commit = (git.stdout.strip() + (" (uncommitted changes)" if dirty else "")) or None
    source["commit"] = commit
    return source


def main():
    args = parse_args(__doc__.splitlines()[1], extra)
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])

    current = {row["filename"] for row in clips(cfg)}
    rows = [r for r in manifest.at("ok", args.speaker) if r["filename"] in current]
    if not rows:
        sys.exit("no verified clips to list")

    ids = {}
    for speaker in dict.fromkeys(r["speaker"] for r in rows):
        ids[speaker] = cfg["speakers"][speaker].get("molana_id")
    unset = [s for s, i in ids.items() if i is None]
    if unset:
        sys.exit(f"set molana_id in the config for: {', '.join(unset)} — the speaker "
                 "column of molana's training list for the same voice")

    listed = []
    for row in rows:
        transcript = molana_transcript(row["phonemes"]).strip()
        foreign = sorted(set(transcript) - TRANSCRIPT_SYMBOLS)
        if foreign:
            reject(row, "symbols molana's transcripts do not use: " + " ".join(foreign))
            continue
        row["transcript"] = transcript
        listed.append(row)
    manifest.save()

    out = cfg["output"] / "molana"
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for row in sorted(listed, key=lambda r: r["filename"]):
        line = f"{row['filename']}|{row['transcript']}|{ids[row['speaker']]}"
        share = int(hashlib.sha1(row["filename"].encode()).hexdigest()[:8], 16) / 16 ** 8
        (val if share < args.val else train).append(line)
        link(converted_wav(cfg, row), wavs / row["filename"])
    names = {r["filename"] for r in listed}
    for stale in wavs.iterdir():
        if stale.name not in names:
            stale.unlink()

    (out / "train_list_en.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val_list_en.txt").write_text("\n".join(val) + ("\n" if val else ""), encoding="utf-8")

    symbols = collections.Counter(c for r in listed for c in r["transcript"] if c in NEW_SYMBOLS)
    per_speaker = collections.Counter(r["speaker"] for r in listed)
    hours = collections.Counter()
    for r in listed:
        hours[r["speaker"]] += float(r["duration_s"]) / 3600
    meta = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vaguye": vaguye_source(),
        "clips": dict(per_speaker),
        "hours": {s: round(h, 2) for s, h in hours.items()},
        "molana_ids": ids,
        "train": len(train),
        "val": len(val),
        "new_symbols": {s: symbols[s] for s in NEW_SYMBOLS},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

    print(f"{out}: {len(train)} train + {len(val)} val lines, "
          + ", ".join(f"{s} {per_speaker[s]} clips / {hours[s]:.2f} h" for s in per_speaker))
    if len(listed) < len(rows):
        print(f"{len(rows) - len(listed)} clips rejected for their symbols, see `note`")
    print("symbols molana has never trained on, rarest first:")
    print("  " + "  ".join(f"{s} {symbols[s]}" for s in sorted(NEW_SYMBOLS, key=lambda s: symbols[s])))


def link(source, target):
    """target is source: a hard link when they share a disk, else a copy."""
    if target.exists():
        if os.path.samefile(source, target):
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


if __name__ == "__main__":
    main()

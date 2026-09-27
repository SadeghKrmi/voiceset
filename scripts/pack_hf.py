#!/usr/bin/env python3
"""Verified clips -> one batch of the molana-tts-audiosets dataset's english/ subset.

    python scripts/pack_hf.py --batch 2026-09-kokoro-seedvc [--out output/hf_english]

Writes into --out, a local mirror of the dataset's english/ folder:

  wavs-<batch>.tar   plain tar holding wavs/<filename>.wav, every verified clip
  metadata.csv       speaker|filename|duration_seconds|text
  phonemes.csv       speaker|filename|phonemes
  scores.csv         speaker|filename|category|kokoro_voice|kokoro_speed|cer|stt_text
  info-card.txt      the batch's sources, settings and counts

The CSVs are appended to, in the other subsets' format (pipe-delimited, no
header, filename without .wav), so seeding --out with the Hub's current files
adds a batch; a filename already in metadata.csv is refused. One tar per batch,
as in transcribed/, so a later batch is one new file on the Hub.

`text` is the English sentence, for reading. The transcript to train on is
phonemes.csv — what Kokoro said, in molana's conventions, as make_list.py
writes it. Phonemizing `text` again would disagree with the audio wherever
misaki and vaguye read a word differently.
"""

import collections
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from common import ROOT, Manifest, converted_wav, load_config, parse_args
from make_list import transcribe, vaguye_source, verified


def extra(parser):
    parser.add_argument("--batch", required=True, help="batch name, used as wavs-<batch>.tar")
    parser.add_argument("--out", help="the english/ mirror to write into (default: <output>/hf_english)")


def clean(text: str) -> str:
    """One field of a pipe-delimited line."""
    return " ".join(text.replace("|", " ").split())


def append(path: Path, lines):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        for fields in lines:
            handle.write("|".join(fields) + "\n")


def voiceset_commit() -> str:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return head or "unknown"


def main():
    args = parse_args(__doc__.splitlines()[1], extra)
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])
    out = Path(args.out) if args.out else cfg["output"] / "hf_english"
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / f"wavs-{args.batch}.tar"
    if tar_path.exists():
        sys.exit(f"{tar_path} already exists: pick a new --batch name")

    rows = transcribe(verified(cfg, manifest, args.speaker))
    manifest.save()
    meta_path = out / "metadata.csv"
    packed = set()
    if meta_path.exists():
        packed = {line.split("|")[1] for line in meta_path.read_text(encoding="utf-8").splitlines() if line}
    rows = sorted((r for r in rows if r["filename"][:-len(".wav")] not in packed),
                  key=lambda r: (r["speaker"], r["filename"]))
    if not rows:
        sys.exit("no new verified clips to pack")

    with tarfile.open(tar_path, "w") as tar:
        for row in tqdm(rows, desc="tar"):
            tar.add(converted_wav(cfg, row), arcname=f"wavs/{row['filename']}")

    def name(row):
        return row["filename"][:-len(".wav")]

    # the manifest keeps two decimals; the other subsets give three, from the audio
    seconds = {r["filename"]: sf.info(str(converted_wav(cfg, r))).duration for r in rows}
    append(meta_path, ([r["speaker"], name(r), f"{seconds[r['filename']]:.3f}", clean(r["text"])]
                       for r in rows))
    append(out / "phonemes.csv", ([r["speaker"], name(r), r["transcript"]] for r in rows))
    append(out / "scores.csv", ([r["speaker"], name(r), r["category"], r["kokoro_voice"],
                                 r["kokoro_speed"], r["cer"], clean(r["stt_text"])] for r in rows))
    card = info_card(cfg, manifest, rows, args.batch, tar_path.name)
    with open(out / "info-card.txt", "a", encoding="utf-8") as handle:
        handle.write(card)

    print(card)
    print(f"{out}: {len(rows)} clips in {tar_path.name} ({tar_path.stat().st_size / 1e9:.2f} GB)")


def info_card(cfg, manifest, rows, batch, tar_name) -> str:
    vc = cfg.get("seed_vc", {})
    verify = cfg.get("verify", {})
    made = [r for r in manifest.rows.values() if r.get("status") in ("ok", "rejected")]
    per_speaker = collections.defaultdict(lambda: [0, 0.0])
    per_category = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        per_speaker[r["speaker"]][0] += 1
        per_speaker[r["speaker"]][1] += float(r["duration_s"])
        per_category[r["category"]][r["speaker"]] += 1
    speakers = sorted(per_speaker)
    vaguye = vaguye_source()

    lines = [f"### Batch {batch}  ({tar_name})", "",
             f"    Built {datetime.now(timezone.utc):%Y-%m-%d} by voiceset {voiceset_commit()} "
             "(github.com/SadeghKrmi/voiceset)",
             "    English sentences spoken by Kokoro-82M, converted to each voice with seed-vc v1",
             "    fine-tunes; 24000 Hz, mono, 16-bit PCM, loudness-normalised."]
    for speaker in speakers:
        spec = cfg["speakers"][speaker]
        speeds = sorted({r["kokoro_speed"] for r in rows if r["speaker"] == speaker})
        voices = sorted({r["kokoro_voice"] for r in rows if r["speaker"] == speaker})
        checkpoint = Path(spec["checkpoint"]).name if spec.get("checkpoint") else "stock zero-shot model"
        lines.append(f"    {speaker}: Kokoro {', '.join(voices)} at speed {', '.join(speeds)}; "
                     f"seed-vc {checkpoint}")
    lines += [f"    seed-vc: diffusion steps {vc.get('diffusion_steps', 30)}, "
              f"cfg rate {vc.get('inference_cfg_rate', 0.7)}, {vc.get('loudness', -18.0)} LUFS",
              f"    Transcripts (phonemes.csv): Kokoro's phonemes in molana's conventions, "
              f"vaguye {vaguye.get('commit') or vaguye.get('version')}",
              f"    Kept only: faster-whisper {verify.get('model', 'large-v3')} hears the sentence "
              f"(CER <= {verify.get('max_cer', 0.05)}), "
              f"{verify.get('min_seconds', 1.0)}-{verify.get('max_seconds', 12.5)} s  "
              f"({len(rows)} of {len(made)} clips)", "",
              f"    {'speaker':12}{'clips':>8}{'hours':>8}"]
    for speaker in speakers:
        count, seconds = per_speaker[speaker]
        lines.append(f"    {speaker:12}{count:8}{seconds / 3600:8.2f}")
    lines += ["", f"    {'category':12}" + "".join(f"{s:>10}" for s in speakers)]
    for category in sorted(per_category):
        lines.append(f"    {category:12}" + "".join(f"{per_category[category][s]:10}" for s in speakers))
    return "\n".join(lines) + "\n\n"


if __name__ == "__main__":
    main()

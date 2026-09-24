#!/usr/bin/env python3
"""Stage 1: sentences -> Kokoro speech, and the phonemes Kokoro said.

    python scripts/synth.py [--speaker ganji] [--limit 20]

Adds a manifest row for every usable sentence that has none yet (see
common.clips for how a sentence gets its speaker and Kokoro voice), then speaks
each row still waiting, into <output>/_kokoro/<speaker>/.

The phonemes are kept from the very tokens Kokoro is given, not worked out from
the text again later, so the transcript make_list.py writes is what is in the
audio — misaki's heteronym choice and espeak-ng's guesses included.
"""

import warnings

import soundfile as sf
from tqdm import tqdm

from common import (SAMPLE_RATE, Manifest, clips, kokoro_speed, kokoro_wav, load_config,
                    parse_args, reject, sentence_problem)

SAVE_EVERY = 50


def extra(parser):
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="default: cuda if available")


def main():
    args = parse_args(__doc__.splitlines()[1], extra)
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])

    skipped = 0
    added = 0
    for row in clips(cfg):
        if sentence_problem(row["text"]):
            skipped += 1                    # lint.py says why
        elif manifest.add(row):
            added += 1
    manifest.save()
    print(f"manifest: {added} new clips, {skipped} sentences skipped (run lint.py for why)")
    warn_other_speeds(cfg, manifest)

    todo = manifest.at("", args.speaker, args.limit)
    if not todo:
        print("nothing to synthesize")
        return

    warnings.filterwarnings("ignore")
    from kokoro import KPipeline
    kokoro = cfg.get("kokoro", {})
    pipeline = KPipeline(lang_code="a", repo_id=kokoro.get("repo_id", "hexgrad/Kokoro-82M"),
                         device=args.device)

    try:
        for n, row in enumerate(tqdm(todo, desc="kokoro"), 1):
            synthesize(pipeline, cfg, row, kokoro_speed(cfg, row["speaker"]))
            if n % SAVE_EVERY == 0:
                manifest.save()
    finally:
        manifest.save()
    done = sum(r["status"] == "synth" for r in todo)
    print(f"synthesized {done} of {len(todo)}; the rest are rejected, see `note`")


def warn_other_speeds(cfg, manifest):
    """Clips already spoken at another speed than the config's are not redone:
    say so, since changing the speed would otherwise seem to have no effect."""
    other = [r for r in manifest.rows.values()
             if r.get("status") not in ("", "rejected")
             and r.get("kokoro_speed") != str(kokoro_speed(cfg, r["speaker"]))]
    if other:
        speakers = sorted({r["speaker"] for r in other})
        print(f"note: {len(other)} clips ({', '.join(speakers)}) were made at another speed "
              "than the config's, or before speeds were recorded. They are kept; to redo "
              "them, delete the output folder (or their rows) and run again.")


def synthesize(pipeline, cfg, row, speed):
    _, tokens = pipeline.g2p(row["text"])
    # Kokoro's G2P drops a word it has no reading for (no lexicon entry and no
    # espeak-ng): the audio would be missing it.
    unsaid = [t.text for t in tokens if any(c.isalpha() for c in t.text) and not t.phonemes]
    if unsaid:
        return reject(row, "no pronunciation for: " + " ".join(unsaid))

    results = list(pipeline.generate_from_tokens(tokens, voice=row["kokoro_voice"], speed=speed))
    if len(results) != 1:
        return reject(row, f"Kokoro split it into {len(results)} parts: shorten it")

    path = kokoro_wav(cfg, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, results[0].audio.cpu().numpy(), SAMPLE_RATE, subtype="PCM_16")
    row["kokoro_speed"] = str(speed)
    row["phonemes"] = results[0].phonemes
    row["status"] = "synth"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where the pipeline stands: clips per stage and speaker, hours kept, why clips were rejected.

    python scripts/status.py
"""

import collections
import re

from common import Manifest, clips, load_config, parse_args, sentence_problem

STAGES = ("", "synth", "converted", "ok", "rejected")
NAMES = {"": "waiting", "synth": "synthesized", "converted": "converted", "ok": "verified",
         "rejected": "rejected"}


def main():
    args = parse_args(__doc__.splitlines()[1])
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])
    rows = [r for r in manifest.rows.values() if not args.speaker or r["speaker"] == args.speaker]

    usable = {r["filename"] for r in clips(cfg) if not sentence_problem(r["text"])}
    unstarted = len(usable - set(manifest.rows))
    if unstarted:
        print(f"{unstarted} sentences have no clip yet: run synth.py")

    speakers = sorted({r["speaker"] for r in rows})
    counts = collections.Counter((r["speaker"], r.get("status", "")) for r in rows)
    print(f"{'':12}" + "".join(f"{NAMES[s]:>13}" for s in STAGES))
    for speaker in speakers:
        print(f"{speaker:12}" + "".join(f"{counts[speaker, s]:>13}" for s in STAGES))

    kept = [r for r in rows if r.get("status") == "ok"]
    if kept:
        hours = sum(float(r["duration_s"]) for r in kept) / 3600
        print(f"\nverified audio: {hours:.2f} h")

    # "cer 0.750 > 0.05" and "cer 0.210 > 0.05" are one reason
    reasons = collections.Counter(re.sub(r"\d+(\.\d+)?", "#", r["note"].split(":")[0])
                                  for r in rows if r.get("status") == "rejected")
    if reasons:
        print("\nrejected because:")
        for reason, n in reasons.most_common():
            print(f"  {n:6}  {reason}")


if __name__ == "__main__":
    main()

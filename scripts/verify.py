#!/usr/bin/env python3
"""Stage 3: does the converted clip still say its sentence?

    python scripts/verify.py [--speaker ganji] [--limit 20]

Voice conversion sometimes slurs a word, drops one, or leaves noise behind, and
a clip like that teaches molana the wrong sound for its transcript. Each
converted clip is transcribed with faster-whisper and compared with the
sentence; a clip is kept (status `ok`) when

  - its character error rate is at most `verify.max_cer`, counted on letters
    only: "Wi-Fi" against "WiFi" or "three-day" against "3-day" is not an
    error, a missing or garbled word is;
  - it lasts between `verify.min_seconds` and `verify.max_seconds` (molana's
    training list stops at 12.5 s).

Everything else is rejected with the reason in `note`; the wav stays on disk to
listen to.
"""

import re
import warnings

import Levenshtein
from num2words import num2words
from tqdm import tqdm

from common import Manifest, converted_wav, load_config, parse_args, reject

SAVE_EVERY = 50


def letters(text: str) -> str:
    """What a sentence says, as bare letters: numbers as words, nothing else."""
    text = text.replace("%", " percent ")        # Whisper writes "twenty percent" as 20%
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 300,000 is one number
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", text)
    return re.sub(r"[^a-z]", "", text.lower())


def cer(reference: str, heard: str) -> float:
    reference, heard = letters(reference), letters(heard)
    return Levenshtein.distance(reference, heard) / max(len(reference), 1)


def main():
    args = parse_args(__doc__.splitlines()[1])
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])
    todo = manifest.at("converted", args.speaker, args.limit)
    if not todo:
        print("nothing to verify")
        return

    settings = cfg.get("verify", {})
    max_cer = settings.get("max_cer", 0.05)
    shortest = settings.get("min_seconds", 1.0)
    longest = settings.get("max_seconds", 12.5)

    warnings.filterwarnings("ignore")
    import torch
    from faster_whisper import WhisperModel
    cuda = torch.cuda.is_available()
    model = WhisperModel(settings.get("model", "large-v3"), device="cuda" if cuda else "cpu",
                         compute_type="float16" if cuda else "int8")

    try:
        for n, row in enumerate(tqdm(todo, desc="whisper"), 1):
            segments, _ = model.transcribe(str(converted_wav(cfg, row)), language="en",
                                           beam_size=5, condition_on_previous_text=False)
            row["stt_text"] = " ".join(s.text.strip() for s in segments)
            error = cer(row["text"], row["stt_text"])
            row["cer"] = f"{error:.3f}"
            seconds = float(row["duration_s"])
            if error > max_cer:
                reject(row, f"cer {error:.3f} > {max_cer}")
            elif not shortest <= seconds <= longest:
                reject(row, f"{seconds:.1f} s, outside {shortest}-{longest} s")
            else:
                row["status"] = "ok"
            if n % SAVE_EVERY == 0:
                manifest.save()
    finally:
        manifest.save()
    print(f"kept {sum(r['status'] == 'ok' for r in todo)} of {len(todo)}")


if __name__ == "__main__":
    main()

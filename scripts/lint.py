#!/usr/bin/env python3
"""Check the sentence files before any GPU time is spent on them.

    python scripts/lint.py

Runs on CPU in seconds, with misaki alone (no Kokoro, no seed-vc), and reports:
  - sentences synth.py will skip, and why (digits, symbols, too short or long)
  - words not in misaki's lexicon: Kokoro still says them, from espeak-ng's
    guess, so listen to a few before trusting them
  - sentences that appear twice (they make one clip)
  - how the clips split between the speakers
  - how often each symbol molana has never trained on turns up — the English
    set is the only place the model learns them, so a rare one wants more
    sentences written for it
"""

import collections
import warnings

from common import (NEW_SYMBOLS, clips, load_config, molana_transcript, parse_args,
                    read_sentences, sentence_problem)


def main():
    args = parse_args(__doc__.splitlines()[1])
    cfg = load_config(args.config)

    warnings.filterwarnings("ignore")
    from misaki import en
    g2p = en.G2P(trf=False, british=False, fallback=None)    # an unknown word: phonemes None

    per_file = collections.Counter()
    rejected, seen, duplicates = [], {}, []
    unknown = collections.Counter()
    symbols = collections.Counter()

    for category, number, text in read_sentences(cfg["sentences"]):
        where = f"{category}.txt:{number}"
        per_file[category] += 1
        problem = sentence_problem(text)
        if problem:
            rejected.append((where, text, problem))
            continue
        if text in seen:
            duplicates.append((where, seen[text]))
            continue
        seen[text] = where
        phonemes, tokens = g2p(text)
        unknown.update(t.text for t in tokens if t.phonemes is None)
        symbols.update(c for c in molana_transcript(phonemes) if c in NEW_SYMBOLS)

    print(f"{sum(per_file.values())} sentences in {cfg['sentences']}")
    for category, count in sorted(per_file.items()):
        print(f"  {category:16} {count}")

    if rejected:
        print(f"\n{len(rejected)} sentences synth.py will skip — fix or remove them:")
        for where, text, problem in rejected:
            print(f"  {where:20} {problem}\n  {'':20} {text}")
    if duplicates:
        print(f"\n{len(duplicates)} duplicates (one clip is made):")
        for where, first in duplicates:
            print(f"  {where} repeats {first}")
    if unknown:
        print(f"\n{len(unknown)} words not in misaki's lexicon — espeak-ng will guess them:")
        print("  " + ", ".join(f"{w} ({n})" if n > 1 else w for w, n in unknown.most_common()))

    unique = {row["filename"]: row["speaker"] for row in clips(cfg)
              if not sentence_problem(row["text"])}
    speakers = collections.Counter(unique.values())
    print(f"\nclips: {sum(speakers.values())} — "
          + ", ".join(f"{s} {n}" for s, n in speakers.items()))

    print("\nsymbols molana has never trained on, rarest first:")
    for symbol in sorted(NEW_SYMBOLS, key=lambda s: symbols[s]):
        print(f"  {symbol}  {symbols[symbol]}")


if __name__ == "__main__":
    main()

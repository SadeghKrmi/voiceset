#!/usr/bin/env python3
"""English and mixed lines for molana's OOD texts.

    python scripts/make_ood.py --english SENTENCES.txt [--mixed MIXED.txt] [--count 500]

In half of its adversarial steps, molana's training (stage 2, and the fine-tune
that adds English) speaks a random line of its OOD texts and has a
discriminator judge the result against real speech: those lines are what the
model learns to say naturally beyond its own training sentences. molana's
OOD_texts.txt is all Persian. This writes, into <output>/molana/:

  OOD_texts_en.txt     English sentences from SENTENCES.txt that are not in the
                       sentence files, phonemized as the training transcripts
                       are: Kokoro's G2P, then common.molana_transcript
  OOD_texts_mixed.txt  lines of MIXED.txt — Persian text with English words,
                       already through molana's text chain (pernorm, Hamnevise,
                       Zirneshane) — phonemized by vaguye with native English,
                       as molana reads them once its engine uses native English

Lines are drawn at random, with a fixed seed, from those that pass: English
sentences by the sentence rules, and every line only if it comes to 50–250
phoneme characters (molana's OOD sampler redraws lines under 50) in symbols
molana's transcripts use.
"""

import contextlib
import io
import random
import warnings

from common import (TRANSCRIPT_SYMBOLS, load_config, molana_transcript, normalize, parse_args,
                    read_sentences, sentence_problem)

# The symbols vaguye's Persian emits, as molana's Persian training lists hold them.
PERSIAN_SYMBOLS = set("abdefhijklmnopqrstuvxzæɒɡʃʒʔˈˌː ,.!?;")
MIN_CHARS, MAX_CHARS = 50, 250


def extra(parser):
    parser.add_argument("--english", required=True, help="English sentences, one per line")
    parser.add_argument("--mixed", help="Persian lines with English words, through molana's text chain")
    parser.add_argument("--count", type=int, default=500, help="lines of each kind (default: 500)")


def fits(phonemes: str, symbols: set) -> bool:
    return MIN_CHARS <= len(phonemes) <= MAX_CHARS and set(phonemes) <= symbols


def english_lines(path, known: set, count: int) -> list:
    warnings.filterwarnings("ignore")
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a", model=False)       # the G2P alone

    sentences = [normalize(line) for line in open(path, encoding="utf-8")]
    random.Random(0).shuffle(sentences)
    lines = []
    for text in sentences:
        if not text or text in known or sentence_problem(text):
            continue
        _, tokens = pipeline.g2p(text)
        if any(any(c.isalpha() for c in t.text) and not t.phonemes for t in tokens):
            continue                        # a word Kokoro has no reading for
        phonemes = molana_transcript(KPipeline.tokens_to_ps(tokens)).strip()
        if fits(phonemes, TRANSCRIPT_SYMBOLS):
            lines.append(phonemes)
            if len(lines) == count:
                break
    return lines


def mixed_lines(path, count: int) -> list:
    with contextlib.redirect_stdout(io.StringIO()):       # vaguye reports what it loads
        from vaguye import PersianPhonemizer
        phonemizer = PersianPhonemizer(english="native")

    texts = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    random.Random(0).shuffle(texts)
    lines = []
    for text in texts:
        with contextlib.redirect_stdout(io.StringIO()):   # and each punctuation mark it keeps
            phonemes = " ".join(phonemizer.phonemize(text).split())
        if fits(phonemes, TRANSCRIPT_SYMBOLS | PERSIAN_SYMBOLS):
            lines.append(phonemes)
            if len(lines) == count:
                break
    return lines


def write(path, lines: list, count: int):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    short = f", only {len(lines)} of {count} asked for passed" if len(lines) < count else ""
    print(f"{path}: {len(lines)} lines{short}")


def main():
    args = parse_args(__doc__.splitlines()[1], extra)
    cfg = load_config(args.config)
    out = cfg["output"] / "molana"
    out.mkdir(parents=True, exist_ok=True)

    known = {text for _, _, text in read_sentences(cfg["sentences"])}
    write(out / "OOD_texts_en.txt", english_lines(args.english, known, args.count), args.count)
    if args.mixed:
        write(out / "OOD_texts_mixed.txt", mixed_lines(args.mixed, args.count), args.count)


if __name__ == "__main__":
    main()

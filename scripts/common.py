"""Shared by every stage: the config, the sentence files, the manifest.

The manifest (`<output>/manifest.csv`) is the pipeline's state. One row is one
clip, and each stage takes the rows at the status it consumes and moves them on:

    synth.py     (new)     -> synth       Kokoro wav in <output>/_kokoro/<speaker>/
    convert.py   synth     -> converted   seed-vc wav in <output>/<speaker>/wavs/
    verify.py    converted -> ok          Whisper still hears the sentence
    any stage              -> rejected    the reason is in `note`

Every stage skips what is already done, so a run that stops is resumed by
running the same command again.

The layout is ttsets' (`filename` and `speaker` columns, clips at
`<speaker>/wavs/<filename>`), so ttsets' filter_by_quality.py runs on this
output unchanged. Columns this code does not know, like the ones that filter
adds, are kept as they are.
"""

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "voiceset.yml"

# Kokoro speaks at 24 kHz, and so is molana trained.
SAMPLE_RATE = 24000

FIELDS = ["filename", "speaker", "category", "text", "kokoro_voice", "kokoro_speed", "phonemes",
          "duration_s", "stt_text", "cer", "transcript", "status", "note"]


# ---------------------------------------------------------------- config


def parse_args(description: str, extra=None) -> argparse.Namespace:
    """The options every stage takes, plus the stage's own (`extra(parser)`)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"config file (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("--speaker", help="only this speaker's clips")
    parser.add_argument("--limit", type=int, help="at most this many clips, for a trial run")
    if extra:
        extra(parser)
    return parser.parse_args()


def load_config(path) -> dict:
    """The config, with every path in it made absolute against the config's folder."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    def resolve(value):
        return None if value is None else (base / Path(value).expanduser()).resolve()

    cfg["output"] = resolve(cfg.get("output", "output"))
    cfg["sentences"] = resolve(cfg.get("sentences", "sentences"))
    if cfg.get("seed_vc", {}).get("repo"):
        cfg["seed_vc"]["repo"] = resolve(cfg["seed_vc"]["repo"])
    for speaker in cfg["speakers"].values():
        for key in ("reference", "checkpoint", "config"):
            speaker[key] = resolve(speaker.get(key))
    return cfg


def kokoro_speed(cfg, speaker: str) -> float:
    """The speaker's Kokoro speed: its own `kokoro_speed`, else `kokoro.speed`, else 1.

    Below 1 is slower. Kokoro's duration model decides what to lengthen (vowels
    and pauses more than consonants), which is why speed is set here and not by
    stretching the audio afterwards.
    """
    own = cfg["speakers"][speaker].get("kokoro_speed")
    return float(own if own is not None else cfg.get("kokoro", {}).get("speed", 1.0))


# ---------------------------------------------------------------- sentences

# Letters, spaces and the punctuation molana's transcripts use. Anything else is
# either something Kokoro reads its own way (digits, symbols) or something
# misaki turns into a symbol the transcripts never have (quotes, brackets).
ALLOWED = re.compile(r"[^A-Za-z ,.?!;:'\-]")
MIN_WORDS, MAX_WORDS = 3, 30


def normalize(line: str) -> str:
    """One sentence line, tidied: single spaces, straight apostrophes."""
    line = line.replace("’", "'").replace("‘", "'")
    return " ".join(line.split())


def read_sentences(directory):
    """(category, line number, text) for every sentence; the file name is the category.

    Blank lines and lines starting with # are skipped.
    """
    for path in sorted(Path(directory).glob("*.txt")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = normalize(line)
            if text and not text.startswith("#"):
                yield path.stem, number, text


def sentence_problem(text: str):
    """Why this sentence cannot be used, or None if it can."""
    if re.search(r"\d", text):
        return "digits: write numbers as words (vaguye leaves digits in English unread)"
    bad = sorted(set(ALLOWED.findall(text)))
    if bad:
        return "characters not allowed: " + " ".join(bad)
    words = len(text.split())
    if words < MIN_WORDS:
        return f"{words} words, fewer than {MIN_WORDS}"
    if words > MAX_WORDS:
        return f"{words} words, more than {MAX_WORDS}: split it"
    return None


def clips(cfg):
    """Every clip the sentence files ask for, as a manifest row (status still empty).

    A clip's name comes from a hash of its text, so it survives the sentence
    moving to another line or file. With `assign: split` the same hash picks
    the one speaker who says it, and the Kokoro voice from that speaker's list;
    `assign: all` gives every sentence to every speaker. Adding a speaker to a
    `split` config reassigns sentences, so settle the speakers first.
    """
    speakers = list(cfg["speakers"])
    every = cfg.get("assign", "split") == "all"
    for category, _, text in read_sentences(cfg["sentences"]):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        number = int(digest[:12], 16)
        for speaker in speakers if every else [speakers[number % len(speakers)]]:
            voices = cfg["speakers"][speaker]["kokoro_voices"]
            yield {
                "filename": f"en_{speaker}_{digest[:10]}.wav",
                "speaker": speaker,
                "category": category,
                "text": text,
                "kokoro_voice": voices[number // len(speakers) % len(voices)],
            }


# ---------------------------------------------------------------- transcripts

# The native-English symbols the Persian training data never has (see vaguye's
# english/native): the English set is what has to teach each of them.
NEW_SYMBOLS = "AIOWYwðŋɑɔəɛɜɪɹɾʊʌʤʧθᵻ"

# Everything a transcript may hold: what vaguye's native English emits, plus
# the punctuation molana's transcripts keep.
TRANSCRIPT_SYMBOLS = set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʔʤʧˈˌːθᵻ ,.!?;:")

# Kokoro v1.0's misaki writes the American flap as T and the article "a" as ɐ.
# vaguye writes ɾ and ə, and a transcript has to use the symbols molana will be
# given at inference. Done before to_molana, which only knows ə as a vowel.
KOKORO_TO_VAGUYE = str.maketrans({"T": "ɾ", "ɐ": "ə"})


def molana_transcript(phonemes: str) -> str:
    """The phonemes Kokoro spoke, written the way molana's transcripts are."""
    from vaguye.languages.english.native.misaki import to_molana
    return to_molana(phonemes.translate(KOKORO_TO_VAGUYE))


# ---------------------------------------------------------------- manifest


class Manifest:
    def __init__(self, output: Path):
        self.path = Path(output) / "manifest.csv"
        self.fields = list(FIELDS)
        self.rows = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.fields += [f for f in reader.fieldnames or [] if f not in self.fields]
                for row in reader:
                    self.rows[row["filename"]] = row

    def add(self, row: dict) -> bool:
        """Add a new clip; False if it is already here."""
        if row["filename"] in self.rows:
            return False
        self.rows[row["filename"]] = {**dict.fromkeys(self.fields, ""), **row}
        return True

    def at(self, status: str, speaker: str = None, limit: int = None):
        """The rows waiting at `status`, optionally one speaker's, at most `limit`."""
        rows = [r for r in self.rows.values()
                if r.get("status", "") == status and (speaker is None or r["speaker"] == speaker)]
        return rows[:limit] if limit else rows

    def save(self):
        """Write the manifest; a crash mid-write leaves the previous one intact."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows.values())
        os.replace(temporary, self.path)


def reject(row: dict, reason: str):
    row["status"] = "rejected"
    row["note"] = reason


def kokoro_wav(cfg, row) -> Path:
    return cfg["output"] / "_kokoro" / row["speaker"] / row["filename"]


def converted_wav(cfg, row) -> Path:
    return cfg["output"] / row["speaker"] / "wavs" / row["filename"]

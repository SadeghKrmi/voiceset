# voiceset — English clips in Ganji's and Parimon's voices, for molana

molana has only ever heard Persian, so 22 of the symbols vaguye's native English
emits (`θ ð w ŋ ɹ ʊ ʌ ɜ`, the diphthongs `A I O W Y`, …) are sounds it has never
made. This builds the English half of a mixed fine-tuning set:

```
sentences/*.txt ──lint.py──▶ (CPU check, before any GPU time)
       │
       ▼ synth.py      Kokoro speaks each sentence; the phonemes it spoke are kept
output/_kokoro/<speaker>/<clip>.wav
       │
       ▼ convert.py    seed-vc v1 (your fine-tunes) turns it into Ganji / Parimon,
                       24 kHz, -18 LUFS
output/<speaker>/wavs/<clip>.wav
       │
       ▼ verify.py     faster-whisper must still hear the sentence
       │
       ▼ make_list.py  transcript = Kokoro's phonemes in molana's conventions
output/molana/{train_list_en.txt, val_list_en.txt, wavs/, meta.json}
```

`output/manifest.csv` holds every clip's state; `python scripts/status.py`
summarizes it. Every stage skips what is done, so an interrupted run resumes by
running the same command again.

The clips are **whole English sentences**, trained *alongside* molana's
Persian data, not spliced into Persian: Kokoro has no Persian, and splicing
audio teaches the model a jump at every language switch. What prepares it for
English inside Persian text is the *content* — the words Persian writing
actually borrows (tech, brands, names, science, loanwords).

## 1. Write sentences

`sentences/<category>.txt`, one sentence per line, `#` for comments. The file
name becomes the `category` column. The starter files show each kind.

- 3–30 words, about 3–12 s of speech
- letters and `, . ? ! ; : ' -` only — **no digits** (write "twenty twenty-four";
  vaguye leaves digits in English unread, so molana would never be asked to say
  them), no quotes, brackets or symbols
- name the English that Persian text borrows; keep `coverage.txt` growing for
  the rarest symbols

Check them — seconds, on CPU:

```bash
python scripts/lint.py
```

It lists sentences that will be skipped and why, duplicates, words misaki does
not know (Kokoro still says them, from espeak-ng's guess — listen to a few),
the split between speakers, and how often each new symbol occurs, rarest first.

Each sentence goes to **one** speaker (`assign: split`), chosen by a hash of its
text, as is the Kokoro voice from that speaker's list — so settle the speakers
before generating: adding one later reassigns sentences.

## 2. Set up the GPU box

```bash
# seed-vc v1 — or reuse the clone and environment you fine-tuned in
git clone https://github.com/Plachtaa/seed-vc /workspace/seed-vc
pip install -r /workspace/seed-vc/requirements.txt

# this project
git clone <voiceset> /workspace/voiceset       # or copy the folder
pip install -r /workspace/voiceset/requirements.txt
```

Then edit `voiceset.yml`:

- `seed_vc.repo` — the clone
- per speaker: `checkpoint` and `config` — your fine-tune's `ft_model.pth` and
  the config `.yml` seed-vc's `train.py` copied into the run folder
- per speaker: `reference` — 5–10 s of clean, neutral speech by that speaker
  (put it in `refs/`). Keep it short: seed-vc fits reference + clip into a 30 s
  window, and a short reference leaves the whole clip in one pass
- per speaker: `molana_id` — the speaker column of molana's training list for
  the same voice. `make_list.py` refuses to run until it is set

## 3. Run

A trial first — then **listen** to `output/<speaker>/wavs/` before the full run:

```bash
cd /workspace/voiceset
python scripts/synth.py   --limit 10
python scripts/convert.py --limit 10
python scripts/verify.py
python scripts/status.py
```

The full run is the same commands without `--limit`. Any stage also takes
`--speaker ganji`.

Optionally, before `make_list.py`, score the clips with ttsets' DNSMOS filter.
The output uses ttsets' layout, so it runs unchanged, on CPU, wherever ttsets is
installed — here, after copying `output/` back:

```bash
/root/ttsets/.venv/bin/python /root/ttsets/scripts/filter_by_quality.py \
    --manifest output/manifest.csv --min-ovrl 3.0          # add --apply to drop them
```

Then (CPU is fine for this one too):

```bash
python scripts/make_list.py
```

## 4. Add to molana

```bash
cp output/molana/wavs/* /root/molana/datasets/resampled/
cat /root/molana/datasets/train_list_max12.5s.txt output/molana/train_list_en.txt \
    > /root/molana/datasets/train_list_mixed.txt
cat /root/molana/datasets/val_list_without_shorts.txt output/molana/val_list_en.txt \
    > /root/molana/datasets/val_list_mixed.txt
```

and point `data_params.train_data` / `val_data` in molana's config at the mixed
lists. Keep the Persian data in: fine-tuning on English alone would pull the
sounds both languages share toward English.

## How the transcripts are made

The transcript is **what Kokoro said**, not what vaguye would say for the text:
`synth.py` keeps the phonemes from the very tokens Kokoro is given, and
`make_list.py` writes them the way molana's transcripts are written
(`common.molana_transcript`):

- Kokoro v1.0's two private symbols become vaguye's: the American flap `T` →
  `ɾ`, the article `ɐ` → `ə`
- then vaguye's `to_molana`: `ᵊ` → `ə`, `i u` → `iː uː`, the stress mark moved
  to the syllable onset

So misaki's heteronym choice ("I **read** it yesterday" → `ɹɛd`) and espeak-ng's
guesses are transcribed as spoken. Where vaguye would say a word differently at
inference, that is vaguye's to fix — the model has still learned the right
sounds for the symbols.

The conventions are vaguye's, so `meta.json` records which vaguye wrote them.
If vaguye changes them, re-run `make_list.py` with the new one: the audio stays,
only the transcripts change.

## Known limits

- `convert.py` has not run on a GPU yet. Everything around it has (Kokoro,
  Whisper, lists, the 24 kHz / loudness step), but its conversion is a port of
  seed-vc's `inference.py` `main()` — models and reference prepared once instead
  of per file — so make the `--limit 10` trial and listen before the full run.
  It follows seed-vc at commit `51383ef`; when updating the clone, diff
  `inference.py`'s `main()` against `SeedVC.convert`.
- seed-vc v1's vocoder runs at 22.05 kHz, so the converted clips carry nothing
  above ~11 kHz, where molana's recordings go to 12 kHz.
- The clips have Kokoro's rhythm and intonation — voice conversion changes the
  voice, not the delivery. Pick Kokoro voices close to each speaker in pitch.

## Tested

On CPU, with the 42 starter clips: `lint.py`; `synth.py` (Kokoro 0.9.4, espeak-ng
fallback for "Samsung" and "Microsoft"); `verify.py` and `make_list.py` with the
Kokoro clips standing in for converted ones — all 42 kept, highest CER 0.021
("Kobe" for COBE), and a clip given another sentence's audio rejected at 0.750;
`convert.py`'s resample and loudness step. seed-vc itself needs the GPU box.

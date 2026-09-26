Builds English fine-tuning data for molana (/root/molana): English sentences →
Kokoro → seed-vc v1 fine-tunes of Ganji and Parimon → verified clips + molana
training lists. README.md is the runbook.

## Layout

- `scripts/common.py` — config loading, sentence files, sentence rules, clip
  naming/assignment, the manifest, transcript conversion. Every stage imports it.
- `scripts/{lint,synth,convert,verify,make_list,status}.py` — one stage each,
  run as `python scripts/<stage>.py [--config] [--speaker] [--limit]`.
- `scripts/tune.py` — seed-vc settings compared on a few sentences, in
  `<output>/_tune/`; outside the pipeline, it never touches the manifest.
- `voiceset.yml` — the config; paths resolve against its folder.
- `sentences/*.txt` — input, one sentence per line, file name = category.
- `output/manifest.csv` — pipeline state; status flows
  `"" → synth → converted → ok`, or `rejected` with the reason in `note`.
  Processes can share it: `Manifest.save` merges only the rows it changed into
  the file, under a lock, so stages run side by side (`--shard i/n`).

## Invariants

- The transcript is what Kokoro spoke: `synth.py` stores the phonemes from the
  tokens it hands Kokoro, and `make_list.py` converts them with
  `common.molana_transcript` (T→ɾ, ɐ→ə, then vaguye's `to_molana`). Never
  re-derive a transcript from the text.
- Output layout stays ttsets' (`filename`, `speaker`, `<speaker>/wavs/<file>`)
  so /root/ttsets/scripts/filter_by_quality.py works on it; unknown manifest
  columns are preserved.
- `convert.py`'s `SeedVC` is a port of seed-vc's `inference.py` `main()` at
  commit 51383ef. Keep it in step with upstream.
- Runtime is the seed-vc v1 environment (torch 2.4.0, transformers 4.46.3,
  numpy 1.26.4) plus `requirements.txt`. Don't let installs move those pins.
- No GPU on this machine: seed-vc runs on a rented GPU box. Kokoro, Whisper
  (small models) and every non-seed-vc stage run on CPU here.

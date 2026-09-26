#!/usr/bin/env python3
"""Try seed-vc settings on a few sentences before the full run.

    python scripts/tune.py [--steps 10 30 50] [--cfg-rates 0.7] [--sentences 8]

Takes the same few sentences for every speaker, one file at a time so the
sample spans the categories, has Kokoro say them once, and converts each clip
with every combination of diffusion steps and cfg rate. The diffusion starts
from the same seeded noise every time, so the settings are the only difference
between the versions of a clip. For each setting it reports

  - seconds per clip, and what that makes the full run cost
  - Whisper's character error rate, counted as verify.py counts it, and how
    many clips verify.py would reject
  - speaker similarity: the cosine between CAM++ embeddings (seed-vc's own
    speaker encoder) of the clip and of the speaker's reference, with Kokoro's
    clip as the baseline to move away from

and writes <output>/_tune/index.html to listen to them side by side, with the
reference first. Nothing here touches the manifest or the pipeline's clips.
"""

import csv
import hashlib
import html
import shutil
import time
import warnings

import numpy as np
import soundfile as sf

from common import SAMPLE_RATE, clips, kokoro_speed, load_config, parse_args, read_sentences, sentence_problem
from convert import SeedVC, finish, import_seed_vc
from synth import synthesize
from verify import cer

SEED = 1234


def extra(parser):
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 30, 50],
                        help="diffusion steps to try (default: 10 30 50)")
    parser.add_argument("--cfg-rates", type=float, nargs="+",
                        help="inference cfg rates to try (default: the config's)")
    parser.add_argument("--sentences", type=int, default=8,
                        help="how many sentences, each said by every speaker (default: 8)")


def sample(cfg, count):
    """`count` sentences, taken in turn from each sentence file; within a file,
    in the order of their hash, so the pick is spread out but always the same."""
    by_file = {}
    for category, _, text in read_sentences(cfg["sentences"]):
        if not sentence_problem(text):
            by_file.setdefault(category, set()).add(text)
    queues = [sorted(texts, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
              for _, texts in sorted(by_file.items())]
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return chosen


def embedding(model, audio, sr):
    """CAM++ speaker embedding, computed the way SeedVC.set_reference does for the reference."""
    import torch
    import torchaudio
    with torch.no_grad():
        wave = torch.tensor(audio).float().unsqueeze(0).to(model.device)
        wave = torchaudio.functional.resample(wave, sr, 16000)
        feat = torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0,
                                                 sample_frequency=16000)
        return model.campplus((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))


def similarity(a, b) -> float:
    import torch
    return torch.nn.functional.cosine_similarity(a, b).item()


def main():
    args = parse_args(__doc__.splitlines()[1], extra)
    cfg = load_config(args.config)
    vc = cfg.get("seed_vc", {})
    cfg_rates = args.cfg_rates or [vc.get("inference_cfg_rate", 0.7)]
    settings = [(steps, rate) for rate in cfg_rates for steps in args.steps]
    label = {s: f"steps{s[0]}_cfg{s[1]:g}" for s in settings}
    tune = cfg["output"] / "_tune"
    speakers = [args.speaker] if args.speaker else list(cfg["speakers"])

    texts = sample(cfg, args.sentences)
    rows = [r for r in clips({**cfg, "assign": "all"})
            if r["text"] in texts and r["speaker"] in speakers]
    print(f"{len(texts)} sentences x {len(speakers)} speakers x {len(settings)} settings")

    warnings.filterwarnings("ignore")
    import torch
    cuda = torch.cuda.is_available()

    # Kokoro, once per clip, into _tune/_kokoro/ (kept between runs)
    at_tune = {**cfg, "output": tune}
    source = {r["filename"]: tune / "_kokoro" / r["speaker"] / r["filename"] for r in rows}
    missing = [r for r in rows if not source[r["filename"]].exists()]
    if missing:
        from kokoro import KPipeline
        kokoro = cfg.get("kokoro", {})
        pipeline = KPipeline(lang_code="a", repo_id=kokoro.get("repo_id", "hexgrad/Kokoro-82M"))
        for row in missing:
            synthesize(pipeline, at_tune, row, kokoro_speed(cfg, row["speaker"]))
        del pipeline
    rows = [r for r in rows if source[r["filename"]].exists()]

    results = {}            # (filename, setting or "kokoro") -> {seconds, similarity, cer, text}
    lufs = vc.get("loudness", -18.0)
    svc = import_seed_vc(vc.get("repo"))
    for speaker in speakers:
        spec = cfg["speakers"][speaker]
        model = SeedVC(svc, spec["checkpoint"], spec["config"], fp16=vc.get("fp16", True))
        model.set_reference(spec["reference"])
        reference = model.reference["style2"]
        (tune / "refs").mkdir(parents=True, exist_ok=True)
        shutil.copy(spec["reference"], tune / "refs" / f"{speaker}.wav")

        mine = [r for r in rows if r["speaker"] == speaker]
        model.convert(source[mine[0]["filename"]], *settings[0])      # warm-up, not timed
        for row in mine:
            path = source[row["filename"]]
            audio, sr = sf.read(path)
            results[row["filename"], "kokoro"] = {
                "similarity": similarity(embedding(model, audio, sr), reference)}
            for setting in settings:
                torch.manual_seed(SEED)
                start = time.perf_counter()
                audio = model.convert(path, *setting)
                if cuda:
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                results[row["filename"], setting] = {
                    "seconds": seconds,
                    "similarity": similarity(embedding(model, audio, model.sr), reference)}
                out = tune / label[setting] / speaker / row["filename"]
                out.parent.mkdir(parents=True, exist_ok=True)
                finished = finish(audio, model.sr, lufs)
                if finished is None:                    # silent: keep a silent file to listen to
                    finished = np.zeros(SAMPLE_RATE, dtype=np.float32)
                sf.write(out, finished, SAMPLE_RATE, subtype="PCM_16")
            print(f"{speaker}: {row['text']}")
        del model
        if cuda:
            torch.cuda.empty_cache()

    # Whisper, as verify.py hears the clips
    from faster_whisper import WhisperModel
    settings_verify = cfg.get("verify", {})
    max_cer = settings_verify.get("max_cer", 0.05)
    whisper = WhisperModel(settings_verify.get("model", "large-v3"), device="cuda" if cuda else "cpu",
                           compute_type="float16" if cuda else "int8")
    for row in rows:
        for setting in ["kokoro", *settings]:
            path = (source[row["filename"]] if setting == "kokoro"
                    else tune / label[setting] / row["speaker"] / row["filename"])
            segments, _ = whisper.transcribe(str(path), language="en", beam_size=5,
                                             condition_on_previous_text=False)
            heard = " ".join(s.text.strip() for s in segments)
            results[row["filename"], setting].update(heard=heard, cer=cer(row["text"], heard))

    report(rows, settings, label, results, speakers, max_cer, cfg)
    write_csv(tune / "results.csv", rows, settings, label, results)
    write_html(tune / "index.html", rows, settings, label, results, speakers, max_cer)
    print(f"\nlisten: {tune / 'index.html'}")


def report(rows, settings, label, results, speakers, max_cer, cfg):
    total = sum(1 for r in clips(cfg) if not sentence_problem(r["text"]))
    head = f"{'setting':18}{'s/clip':>8}{'full run':>10}{'CER mean':>10}{'CER max':>9}{'rejected':>10}"
    head += "".join(f"{'sim ' + s:>14}" for s in speakers)
    print("\n" + head)
    for setting in ["kokoro", *settings]:
        cells = [results[r["filename"], setting] for r in rows]
        errors = [c["cer"] for c in cells]
        if setting == "kokoro":
            line = f"{'kokoro (source)':18}{'':>8}{'':>10}"
        else:
            per_clip = np.mean([c["seconds"] for c in cells])
            line = f"{label[setting]:18}{per_clip:8.2f}{per_clip * total / 3600:9.1f}h"
        line += f"{np.mean(errors):10.3f}{max(errors):9.3f}{sum(e > max_cer for e in errors):10}"
        for speaker in speakers:
            sims = [results[r["filename"], setting]["similarity"] for r in rows if r["speaker"] == speaker]
            line += f"{np.mean(sims):14.3f}"
        print(line)
    print(f"\nfull run = {total} clips at that speed; rejected = CER above {max_cer}; "
          "sim = cosine to the reference speaker (higher is closer)")


def write_csv(path, rows, settings, label, results):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "speaker", "text", "setting", "seconds", "similarity", "cer", "heard"])
        for row in rows:
            for setting in ["kokoro", *settings]:
                c = results[row["filename"], setting]
                writer.writerow([row["filename"], row["speaker"], row["text"],
                                 "kokoro" if setting == "kokoro" else label[setting],
                                 f"{c['seconds']:.3f}" if "seconds" in c else "",
                                 f"{c['similarity']:.3f}", f"{c['cer']:.3f}", c["heard"]])


def write_html(path, rows, settings, label, results, speakers, max_cer):
    def player(src, cell=None):
        note = ""
        if cell is not None:
            flag = ' class="bad"' if cell["cer"] > max_cer else ""
            note = (f'<div{flag}>CER {cell["cer"]:.3f} · sim {cell["similarity"]:.2f}</div>'
                    f'<div class="heard">{html.escape(cell["heard"])}</div>')
        return f'<td><audio controls preload="none" src="{src}"></audio>{note}</td>'

    columns = ["kokoro (source)", *(label[s] for s in settings)]
    out = ["<!doctype html><meta charset=utf-8><title>seed-vc settings</title><style>",
           "body{font:14px system-ui,sans-serif;margin:16px}table{border-collapse:collapse}",
           "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}audio{width:220px}",
           ".heard{color:#666;font-size:12px;max-width:220px}.bad{color:#b00;font-weight:600}",
           "</style><h1>seed-vc settings</h1>"]
    for speaker in speakers:
        out.append(f"<h2>{speaker}</h2><p>reference: <audio controls preload=none "
                   f'src="refs/{speaker}.wav"></audio></p><table><tr><th>sentence</th>')
        out += [f"<th>{html.escape(c)}</th>" for c in columns]
        out.append("</tr>")
        for row in (r for r in rows if r["speaker"] == speaker):
            name = row["filename"]
            out.append(f"<tr><td>{html.escape(row['text'])}</td>")
            out.append(player(f"_kokoro/{speaker}/{name}", results[name, "kokoro"]))
            out += [player(f"{label[s]}/{speaker}/{name}", results[name, s]) for s in settings]
            out.append("</tr>")
        out.append("</table>")
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()

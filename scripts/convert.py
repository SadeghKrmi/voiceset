#!/usr/bin/env python3
"""Stage 2: Kokoro speech -> the target speaker's voice, with seed-vc v1.

    python scripts/convert.py [--speaker ganji] [--limit 20]

seed-vc is not a package. This imports inference.py from a clone of its repo
(`seed_vc.repo` in the config) and calls its load_models(), which is what reads
a fine-tuned checkpoint and config. inference.py's main() converts one file per
process and loads every model again each time; `SeedVC` below is that function's
body with the models, and the reference speaker's side of the conversion,
prepared once. It follows seed-vc at commit 51383ef — when updating the clone,
diff inference.py's main() against `SeedVC.convert`.

Each converted clip is resampled to 24 kHz (molana's rate), brought to the
loudness in `seed_vc.loudness` (ttsets chunks molana's recordings to -18 LUFS),
and written as 16-bit PCM to <output>/<speaker>/wavs/.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from common import (SAMPLE_RATE, Manifest, converted_wav, kokoro_wav, load_config,
                    parse_args, reject)

SAVE_EVERY = 25
PEAK = 0.98             # after loudness normalization, never above this

# inference.py's own settings for one conversion, not exposed by it
MAX_REFERENCE_S = 25
OVERLAP_FRAMES = 16


def import_seed_vc(repo):
    """seed-vc's inference module. It resolves its model cache and configs
    against the working directory, so the working directory moves into the repo."""
    if repo is None or not (repo / "inference.py").exists():
        sys.exit(f"seed_vc.repo must be a clone of github.com/Plachtaa/seed-vc (got {repo})")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import inference
    return inference


class SeedVC:
    def __init__(self, svc, checkpoint, config, fp16: bool):
        """One loaded model. No checkpoint means seed-vc's stock zero-shot model."""
        import torch
        self.torch = torch
        self.svc = svc
        self.device = svc.device
        self.fp16 = fp16
        self.f0_condition = False
        if config is not None:
            model_params = yaml.safe_load(open(config))["model_params"]
            self.f0_condition = bool(model_params.get("DiT", {}).get("f0_condition", False))
        args = argparse.Namespace(checkpoint=checkpoint and str(checkpoint),
                                  config=config and str(config),
                                  fp16=fp16, f0_condition=self.f0_condition)
        (self.model, self.semantic_fn, self.f0_fn, self.vocoder_fn,
         self.campplus, self.to_mel, mel_args) = svc.load_models(args)
        self.sr = mel_args["sampling_rate"]
        self.hop = mel_args["hop_size"]
        self.reference = None

    def set_reference(self, path):
        """The target speaker's side of every conversion: computed once per speaker."""
        import librosa
        import torchaudio
        torch = self.torch
        with torch.no_grad():
            audio = librosa.load(str(path), sr=self.sr)[0][: self.sr * MAX_REFERENCE_S]
            ref = torch.tensor(audio).unsqueeze(0).float().to(self.device)
            ref_16k = torchaudio.functional.resample(ref, self.sr, 16000)
            semantic = self.semantic_fn(ref_16k)
            mel2 = self.to_mel(ref.float())
            feat2 = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0,
                                                      sample_frequency=16000)
            style2 = self.campplus((feat2 - feat2.mean(dim=0, keepdim=True)).unsqueeze(0))
            f0 = None
            if self.f0_condition:
                f0 = torch.from_numpy(self.f0_fn(ref_16k[0], thred=0.03)).to(self.device)[None]
            prompt = self.model.length_regulator(
                semantic, ylens=torch.LongTensor([mel2.size(2)]).to(self.device),
                n_quantizers=3, f0=f0)[0]
        self.reference = {"mel2": mel2, "style2": style2, "f0": f0, "prompt": prompt}

    def convert(self, path, diffusion_steps: int, cfg_rate: float) -> np.ndarray:
        """One clip in the reference speaker's voice, at self.sr."""
        import librosa
        import torchaudio
        torch = self.torch
        ref = self.reference
        with torch.no_grad():
            source = torch.tensor(librosa.load(str(path), sr=self.sr)[0])
            source = source.unsqueeze(0).float().to(self.device)
            source_16k = torchaudio.functional.resample(source, self.sr, 16000)
            if source_16k.size(-1) > 16000 * 30:
                # inference.py handles longer input in overlapping windows;
                # molana's clips never get there.
                raise ValueError("longer than 30 s")
            semantic = self.semantic_fn(source_16k)
            mel = self.to_mel(source.float())

            f0 = None
            if self.f0_condition:
                # auto_f0_adjust: move Kokoro's pitch to the reference speaker's
                f0_alt = torch.from_numpy(self.f0_fn(source_16k[0], thred=0.03)).to(self.device)[None]
                voiced = f0_alt > 1
                log_f0 = torch.log(f0_alt + 1e-5)
                median_ref = torch.median(torch.log(ref["f0"][ref["f0"] > 1] + 1e-5))
                log_f0[voiced] = log_f0[voiced] - torch.median(log_f0[voiced]) + median_ref
                f0 = torch.exp(log_f0)

            cond = self.model.length_regulator(
                semantic, ylens=torch.LongTensor([mel.size(2)]).to(self.device),
                n_quantizers=3, f0=f0)[0]

            mel2 = ref["mel2"]
            max_source_window = self.sr // self.hop * 30 - mel2.size(2)
            overlap_wave = OVERLAP_FRAMES * self.hop
            chunks, previous, processed = [], None, 0
            while processed < cond.size(1):
                chunk = cond[:, processed:processed + max_source_window]
                last = processed + max_source_window >= cond.size(1)
                condition = torch.cat([ref["prompt"], chunk], dim=1)
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.float16 if self.fp16 else torch.float32):
                    target = self.model.cfm.inference(
                        condition, torch.LongTensor([condition.size(1)]).to(self.device),
                        mel2, ref["style2"], None, diffusion_steps,
                        inference_cfg_rate=cfg_rate)
                    target = target[:, :, mel2.size(-1):]
                wave = self.vocoder_fn(target.float()).squeeze()[None, :]
                if previous is None and last:
                    chunks.append(wave[0].cpu().numpy())
                    break
                if previous is None:
                    chunks.append(wave[0, :-overlap_wave].cpu().numpy())
                elif last:
                    chunks.append(self.svc.crossfade(previous.cpu().numpy(),
                                                     wave[0].cpu().numpy(), overlap_wave))
                    break
                else:
                    chunks.append(self.svc.crossfade(previous.cpu().numpy(),
                                                     wave[0, :-overlap_wave].cpu().numpy(),
                                                     overlap_wave))
                previous = wave[0, -overlap_wave:]
                processed += target.size(2) - OVERLAP_FRAMES
        return np.concatenate(chunks)


def finish(audio: np.ndarray, sr: int, lufs: float) -> np.ndarray:
    """At molana's sample rate and the target loudness, and never clipping.
    None if the clip is silent (no loudness to measure)."""
    import librosa
    import pyloudnorm
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    measured = pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(audio)
    if not np.isfinite(measured):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # pyloudnorm warns of clipping; handled below
        audio = pyloudnorm.normalize.loudness(audio, measured, lufs)
    peak = np.abs(audio).max()
    return audio * (PEAK / peak) if peak > PEAK else audio


def main():
    args = parse_args(__doc__.splitlines()[1])
    cfg = load_config(args.config)
    manifest = Manifest(cfg["output"])
    todo = manifest.at("synth", args.speaker, args.limit, args.shard)
    if not todo:
        print("nothing to convert")
        return

    vc = cfg.get("seed_vc", {})
    steps = vc.get("diffusion_steps", 30)
    cfg_rate = vc.get("inference_cfg_rate", 0.7)
    lufs = vc.get("loudness", -18.0)

    warnings.filterwarnings("ignore")
    svc = import_seed_vc(vc.get("repo"))
    models = {}             # one per distinct checkpoint: two speakers may share one

    try:
        for speaker in dict.fromkeys(row["speaker"] for row in todo):
            spec = cfg["speakers"][speaker]
            if spec["reference"] is None or not spec["reference"].exists():
                sys.exit(f"{speaker}: reference wav not found ({spec['reference']})")
            key = (spec["checkpoint"], spec["config"])
            if key not in models:
                models.clear()      # free the last speaker's model before loading the next
                models[key] = SeedVC(svc, *key, fp16=vc.get("fp16", True))
            model = models[key]
            model.set_reference(spec["reference"])

            rows = [r for r in todo if r["speaker"] == speaker]
            for n, row in enumerate(tqdm(rows, desc=f"seed-vc {speaker}"), 1):
                audio = finish(model.convert(kokoro_wav(cfg, row), steps, cfg_rate),
                               model.sr, lufs)
                if audio is None:
                    reject(row, "silent after conversion")
                    continue
                path = converted_wav(cfg, row)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
                row["duration_s"] = f"{len(audio) / SAMPLE_RATE:.2f}"
                row["status"] = "converted"
                if n % SAVE_EVERY == 0:
                    manifest.save()
    finally:
        manifest.save()
    print(f"converted {sum(r['status'] == 'converted' for r in todo)} of {len(todo)}")


if __name__ == "__main__":
    main()

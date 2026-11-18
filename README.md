# Target Speaker ASR (modular, local)

Transcribes **only the enrolled person**, ignoring other speakers and noise. Runs on a
PC/laptop/Mac; a tablet is the microphone + UI over your local network. No audio leaves your network.

```
Tablet browser (mic, enroll, model picker)
   │  16 kHz PCM over WebSocket (HTTPS)
   ▼
VAD ─► utterance segmenter ─► [denoiser] ─► speaker encoder ─► cosine vs. enrolled voice
                                                                  │ match?
                                                                  ▼
                                                                 STT ─► transcript
```

Every box is a swappable **backend**. Nothing is hard-wired to Whisper, Silero or ECAPA.

> Status: the code has been written but not yet executed on real hardware. The GPU paths for AMD, Intel and
> Apple in particular are untested - that is exactly why `device: auto` probes each device and falls back.

## Install with uv

Install [uv](https://docs.astral.sh/uv/) once (`winget install --id=astral-sh.uv -e` on Windows, `brew install uv` on macOS,
or `pip install uv`). Then pick **one hardware extra** - it decides which PyTorch build is installed:

| Your machine | Command |
|---|---|
| NVIDIA GPU, driver supports CUDA 12.8 (check `nvidia-smi`; **required for RTX 50-series**) | `uv sync --extra cu128` |
| NVIDIA GPU, older driver | `uv sync --extra cu126` |
| AMD GPU on Linux (ROCm) | `uv sync --extra rocm` |
| Intel GPU (Arc / Iris Xe), Windows or Linux | `uv sync --extra xpu` |
| Apple Silicon (M1/M2/M3/M4) | `uv sync --extra apple` (+ `--extra mlx` for Whisper on the Apple GPU) |
| No GPU / AMD or Intel on Windows / not sure | `uv sync --extra cpu` |

Add optional backends by listing more extras **in the same command** - `uv sync` is exact and removes anything you leave out:

```powershell
uv sync --extra cu128 --extra funasr --extra hf          # + SenseVoice/Paraformer + MMS/HF models
```

**Install every backend** (so every engine in the web UI works). Pick your hardware extra and add all optional ones:

```bash
# Windows / Linux, NVIDIA (use cu126 for older drivers)
uv sync --extra cu128 --extra funasr --extra hf --extra sherpa
# Apple Silicon Mac (adds mlx_whisper)
uv sync --extra apple --extra funasr --extra hf --extra sherpa --extra mlx
# No GPU
uv sync --extra cpu --extra funasr --extra hf --extra sherpa
```

If the UI shows `No module named 'xxx'` when you pick an engine, its extra is not installed:
`transformers` -> `hf`, `funasr` -> `funasr`, `sherpa_onnx` -> `sherpa`, `mlx_whisper` -> `mlx`.
Re-run `uv sync` with the full list of extras you want (it removes anything left out).

Extras: `funasr`, `hf`, `sherpa`, `mlx` (Apple only). `onnxruntime` is a core dependency (Silero VAD uses it). `deepfilternet` isn't an extra:
`uv pip install deepfilternet`, then run with `uv run --no-sync …`.

## Run

```powershell
cd D:\RG\target_speaker_asr
uv run python scripts/make_cert.py      # HTTPS cert - the tablet microphone requires HTTPS
uv run python run_server.py
```

On the tablet (same Wi-Fi) open `https://<PC-LAN-IP>:8000`, accept the certificate warning once, then:

1. **Target voice** → type a profile id, tap *Record voice* (10–30 s, only that person, quiet), or upload a file.
2. **Models** → choose STT / speaker encoder / VAD. Under each dropdown the UI shows **which device the model runs on**.
3. **Start listening**. Matching speech is transcribed; other speakers appear greyed out if "show rejected" is on.

PC only (no tablet): remove the `ssl_*` lines in `config.yaml` and use `http://localhost:8000`.
Allow inbound port 8000 in Windows Firewall for the tablet. Set `server.token` and open `https://…:8000/?token=<value>` on shared networks.

## OpenAI-compatible API (for chat apps)

A second entry point that exposes the same target-speaker pipeline as `POST /v1/audio/transcriptions`,
so any app with a "custom OpenAI STT endpoint" setting (Open WebUI, Cherry Studio, …) can use it as its
microphone. Audio from other speakers is dropped on the server; the client never knows.

```powershell
uv run python run_server_openai.py --profile alice          # http://<PC-IP>:8001/v1
```

* Enroll the voice first with the web UI (`run_server.py`) or `scripts/cli.py enroll`. Profiles are shared on disk.
* **Client settings:** base URL `http://<PC-IP>:8001/v1`, API key = `server.token` (anything if no token), model = `tsasr-alice`
  (or `alice`). Any other model name (e.g. `whisper-1`) uses `--profile` / `openai.default_profile`. `GET /v1/models` lists profiles.
* `response_format`: `json`, `text`, `verbose_json` (adds per-segment scores and the rejected segments). Not streaming: the
  client uploads a finished clip. Browser-recorded clips (webm/opus) need `ffmpeg` on the PATH.
* Plain HTTP by default (chat backends call it server-to-server; the browser mic permission belongs to the chat page). Use `--ssl`
  to serve the certificate from `scripts/make_cert.py`. Port, default profile and concurrency: `openai:` in `config.yaml`.
* It loads its own copy of the models, so running it next to `run_server.py` uses roughly twice the (V)RAM. Stop one if memory is tight.

## Hardware auto-detection

`device: auto` (the default in every preset) tries, in order **NVIDIA → AMD → Intel → Apple → CPU**.
For each candidate it loads the model *and* runs a 1-second dummy inference; if anything fails (missing DLL, unsupported
operator, out of memory) it logs the reason and moves to the next device. Forcing a device (`device: cuda:1`, `cpu`, `mps`, `xpu`)
disables the fallback so errors are visible. The startup log prints what was detected.

What each backend can actually accelerate:

| Backend | NVIDIA | AMD | Intel GPU | Apple Silicon |
|---|---|---|---|---|
| `faster_whisper` (CTranslate2) | ✅ CUDA | ❌ CPU | ❌ CPU | ❌ CPU (int8, still decent) |
| `mlx_whisper` | – | – | – | ✅ Apple GPU |
| `speechbrain_ecapa`, `funasr`, `hf_transformers` (PyTorch) | ✅ `cuda` | ✅ ROCm (Linux) | ✅ `xpu` | ⚠️ `mps` (some ops may be unsupported → falls back to CPU) |
| `openai_http` + [whisper.cpp](https://github.com/ggml-org/whisper.cpp) server | ✅ | ✅ Vulkan | ✅ Vulkan | ✅ Metal |
| `sherpa_onnx` | CUDA needs the separate `+cuda` sherpa-onnx wheel | CPU | CPU | CPU |
| `silero` VAD | CPU always, via single-thread ONNX. (The TorchScript variant is far slower on many-core CPUs: PyTorch spreads its tiny ops over every core.) | | | |

Practical recommendations:

* **NVIDIA:** the default Whisper preset just works (float16 on GPU). If the log says CUDA failed and it fell back to CPU,
  the usual cause is a missing cuBLAS/cuDNN DLL for CTranslate2: `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
  (then `uv run --no-sync …`).
* **Apple Silicon:** use `mlx_whisper_small` / `mlx_whisper_large_v3` for Whisper on the GPU; PyTorch components use `mps` with CPU fallback.
* **AMD or Intel on Windows:** PyTorch GPU support there is limited, so the simplest big win is running **whisper.cpp built with Vulkan**
  as a local server and selecting the `local_http_server` STT preset. Speaker encoder + VAD are light and run fine on CPU.
* DirectML is intentionally not included (`torch-directml` lags PyTorch and many operators are unsupported).

## Built-in backends

| Kind | Name | Notes |
|---|---|---|
| vad | `silero`, `energy` | Silero is the recommended one; `energy` has no dependencies |
| speaker | `speechbrain_ecapa`, `sherpa_onnx` | sherpa: 3D-Speaker / WeSpeaker / TitaNet `.onnx` models |
| stt | `faster_whisper` | ~99 languages |
| stt | `mlx_whisper` | Whisper on Apple GPU |
| stt | `funasr` | SenseVoice (zh/yue/en/ja/ko), Paraformer |
| stt | `sherpa_onnx` | lightest CPU option; SenseVoice / Paraformer / Whisper / Zipformer / NeMo |
| stt | `hf_transformers` | **bring your own model** - any HF ASR model, e.g. **MMS-1B-all (1100+ languages)**, Whisper fine-tunes, wav2vec2 (see below; no presets) |
| stt | `openai_http` | any local server with `/v1/audio/transcriptions` (whisper.cpp, speaches…) |
| denoiser | `none`, `deepfilternet` | denoising can *hurt* ASR – A/B test before enabling |

Pick the STT per language (Whisper is weak on some): in the UI dropdown, or `use.stt` in `config.yaml`.

## Bring your own model (`hf_transformers`)

For languages the preset engines handle badly, choose **hf_transformers** in the *Speech-to-text engine* dropdown.
It has no presets: nothing is downloaded until you choose a model. Then either

* **type** a Hugging Face id (`facebook/mms-1b-all`, `openai/whisper-large-v3`, your own fine-tune `user/model`)
  or a model **folder on the PC** (`D:\models\my-asr`), then press Enter; or
* **upload** a model from the tablet: zip the Hugging Face model folder itself (`config.json`, weights,
  tokenizer / preprocessor files) and pick the `.zip`. It is unpacked to `models/custom/<name>/` and loaded straight away; or
* tap an entry under **Previously used** - every model that loaded successfully is remembered (shared by all
  tablets/browsers, stored in `models/model_history.json`; `×` forgets an entry but keeps the files).

Notes: pick the device next to the model (`auto` falls back to CPU if the model can't run on the GPU). The language hint
goes to the model as-is (MMS needs ISO-639-3 codes like `eng`, `zsm`, `cmn`; Whisper-style models use `en`, `ms`, `ta` …).

**Security:** the browser may only choose the *model* and *device*. Every other engine option (`trust_remote_code`,
`compute_type`, server addresses, file paths …) comes from `config.yaml`, so a client cannot inject code-execution
options. Even so, model files can contain pickled code - only load models you trust, and set `server.token` if
other people can reach the server (an upload endpoint is open to anyone who can open the page).

## Offline / command line

```powershell
uv run python scripts/cli.py enroll alice sample.wav
uv run python scripts/cli.py transcribe alice meeting.wav --language en --show-rejected
```

## Tuning

| Symptom | Change |
|---|---|
| Target gets rejected | lower `similarity_threshold` (UI slider), enroll more/cleaner audio, `short_policy: tile` |
| Other people leak through | raise `similarity_threshold`; enroll with a mic/room similar to real use |
| Too slow to respond | lower `min_silence_ms` (e.g. 350) and/or use a smaller STT |
| Sentences chopped | raise `min_silence_ms`, raise `max_utterance_s` |
| Noisy environment | try `denoise_for_verify` / `denoise_for_stt` with `deepfilternet`, compare |
| Wrong language | set the language hint (MMS needs ISO-639-3 codes like `eng`, `zsm`) |

Typical ECAPA cosine scores: same speaker ≈ 0.5–0.8, different speakers ≈ 0.0–0.3. The *rejected* lines in the UI
show scores, which makes threshold tuning easy. Switching the speaker encoder re-embeds stored enrollment audio automatically.

## Add your own backend

Create `my_backends.py` in the project root:

```python
import numpy as np
from tsasr.base import STT, Transcript
from tsasr.device import build_with_fallback, probe_noise   # optional: reuse device auto-detect
from tsasr.registry import register

@register("stt", "my_engine")
class MyEngine(STT):
    def __init__(self, model_path: str, device: str = "auto"):
        def build(dev):
            self.device = dev
            self.model = ...           # load your model on `dev` (import heavy libs here)
            return self
        build_with_fallback(build, device, "torch", lambda _: self.transcribe(probe_noise()))

    def transcribe(self, audio: np.ndarray, language=None) -> Transcript:
        # audio: mono float32, 16 kHz
        return Transcript(text="...", language=language)
```

Then in `config.yaml`:

```yaml
plugins: [my_backends]
presets:
  stt:
    mine: {label: "My engine", name: my_engine, params: {model_path: models/foo}}
use:
  stt: mine
```

Same pattern for `vad` (`speech_prob(frame)`), `speaker` (`embed(audio)`) and `denoiser` (`enhance(audio)`) – see `tsasr/base.py`.

## Known limits

* **Overlapping speech**: if the target and someone else talk at the same moment, the utterance embedding is a mix and may
  be rejected, or accepted together with the other voice. A target-speaker *extraction* model (WeSep, SpeakerBeam) can be added as a
  new pre-processing component for that case.
* Verification needs ~1 s of speech, so very short utterances ("yes") are less reliable (`short_policy`).
* Latency = utterance length + `min_silence_ms` + STT time (near-real-time, not word-by-word streaming).

## Layout

```
pyproject.toml        uv project + hardware extras    config.yaml     presets + defaults
tsasr/device.py       auto-detect + fallback          tsasr/base.py   component interfaces
tsasr/registry.py     plugin registry                 tsasr/pipeline.py  VAD segmenter + verify + STT
tsasr/backends/       built-in backends               tsasr/engine.py    hot-swap, enrollment, sessions
web/index.html        tablet UI                       tsasr/server.py    FastAPI + WebSocket
scripts/              make_cert.py, cli.py            run_server.py   entry point
```

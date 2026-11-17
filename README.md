# Target Speaker ASR (modular, local)

Transcribes **only the enrolled person**, ignoring other speakers and noise. Runs on a
PC/laptop; a tablet is the microphone + UI over your local network. No audio leaves your network.

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

## Built-in backends

| Kind | Name | Notes / install |
|---|---|---|
| vad | `silero` | `silero-vad` (torch or onnxruntime) |
| vad | `energy` | no dependencies, quiet rooms only |
| speaker | `speechbrain_ecapa` | `speechbrain` |
| speaker | `sherpa_onnx` | `sherpa-onnx` + 3D-Speaker / WeSpeaker / TitaNet `.onnx` |
| stt | `faster_whisper` | ~99 languages. `faster-whisper` |
| stt | `funasr` | SenseVoice (zh/yue/en/ja/ko), Paraformer. `funasr modelscope` |
| stt | `sherpa_onnx` | lightest CPU option; SenseVoice / Paraformer / Whisper / Zipformer / NeMo. `sherpa-onnx` |
| stt | `hf_transformers` | any Hugging Face ASR model, e.g. **MMS-1B-all (1100+ languages)**, Whisper fine-tunes, wav2vec2. `transformers` |
| stt | `openai_http` | any local server with `/v1/audio/transcriptions` (whisper.cpp, speaches…) |
| denoiser | `none`, `deepfilternet` | denoising can *hurt* ASR – A/B test before enabling |

Pick the STT per language (Whisper is weak on some): in the UI dropdown, or `use.stt` in `config.yaml`.

## Quick start (Windows PowerShell)

```powershell
cd D:\RG\target_speaker_asr
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/make_cert.py          # HTTPS cert (tablet mic requires HTTPS)
python run_server.py
```

On the tablet (same Wi-Fi), open `https://<PC-LAN-IP>:8000`, accept the certificate warning once, and:

1. **Target voice** → type a profile id, tap *Record voice* (10–30 s, only that person, quiet), or upload a file.
2. **Models** → choose STT / speaker encoder / VAD (models download on first use).
3. **Start listening**. Matching speech is transcribed; others appear greyed out if "show rejected" is on.

Without certificate (PC only): remove the `ssl_*` lines in `config.yaml` and use `http://localhost:8000`.
Allow inbound port 8000 in Windows Firewall for the tablet to connect. Set `server.token` and open
`https://…:8000/?token=<value>` if the network is shared.

Models run on CPU by default (`device: cpu`). To use a GPU later, set `device: cuda` (and `compute_type: float16`
for faster-whisper) in the relevant preset.

## Offline / command line

```powershell
python scripts/cli.py enroll alice sample.wav
python scripts/cli.py transcribe alice meeting.wav --language en --show-rejected
```

## Tuning

| Symptom | Change |
|---|---|
| Target gets rejected | lower `similarity_threshold` (UI slider), enroll more/cleaner audio, `short_policy: tile` |
| Other people leak through | raise `similarity_threshold`; enroll with a different mic condition matching use |
| Too slow to respond | lower `min_silence_ms` (e.g. 350) and use a smaller STT |
| Sentences chopped | raise `min_silence_ms`, raise `max_utterance_s` |
| Noisy environment | try `denoise_for_verify` / `denoise_for_stt` with `deepfilternet`, compare |
| Wrong language | set the language hint (MMS needs ISO-639-3 codes like `eng`, `zsm`) |

Typical ECAPA cosine scores: same speaker ≈ 0.5–0.8, different speakers ≈ 0.0–0.3. The *rejected* lines in the UI
show scores, which makes threshold tuning easy. Switching the speaker encoder re-embeds stored enrollment audio automatically.

## Add your own backend

Create `my_backends.py` anywhere importable (e.g. project root):

```python
import numpy as np
from tsasr.base import STT, Transcript
from tsasr.registry import register

@register("stt", "my_engine")
class MyEngine(STT):
    def __init__(self, model_path: str):
        ...                                # load your model (import heavy libs here)
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
  be rejected or accepted along with the other voice. A target-speaker *extraction* model (WeSep, SpeakerBeam) can be added as a
  new pre-processing component for that case.
* Verification needs ~1 s of speech, so very short utterances ("yes") are less reliable (`short_policy`).
* Latency = utterance length + `min_silence_ms` + STT time (near-real-time, not word-by-word streaming).

## Layout

```
config.yaml           presets + defaults        run_server.py   entry point
tsasr/base.py         component interfaces      tsasr/pipeline.py  VAD segmenter + verify + STT
tsasr/registry.py     plugin registry           tsasr/engine.py    hot-swap, enrollment, sessions
tsasr/backends/       built-in backends         tsasr/server.py    FastAPI + WebSocket
web/index.html        tablet UI                 scripts/           make_cert.py, cli.py
```

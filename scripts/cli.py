"""Command-line helpers that use the same engine as the server.

    uv run python scripts/cli.py enroll alice voice_sample.wav
    uv run python scripts/cli.py transcribe alice meeting.m4a [--language zh] [--threshold 0.55]

Progress and model/device info are logged to stderr; transcripts go to stdout as they are produced.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tsasr.audio import load_audio_bytes  # noqa: E402
from tsasr.base import SAMPLE_RATE  # noqa: E402
from tsasr.config import load_config  # noqa: E402
from tsasr.engine import Engine  # noqa: E402

log = logging.getLogger("cli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("profile")
    e.add_argument("audio")
    e.add_argument("--append", action="store_true")

    t = sub.add_parser("transcribe")
    t.add_argument("profile")
    t.add_argument("audio")
    t.add_argument("--language", default=None)
    t.add_argument("--threshold", type=float, default=None)
    t.add_argument("--show-rejected", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    engine = Engine(load_config(args.config))

    t0 = time.perf_counter()
    audio = load_audio_bytes(Path(args.audio).read_bytes())
    log.info("decoded %s: %.1fs of audio (%.1fs to decode)", args.audio, audio.size / SAMPLE_RATE,
             time.perf_counter() - t0)

    if args.cmd == "enroll":
        log.info("enrolling (VAD -> speaker embedding)...")
        print(engine.enroll(args.profile, audio, append=args.append))
        return

    pipe = engine.new_session(args.profile, args.language)  # loads speaker + STT models, logs devices
    if args.threshold is not None:
        pipe.update(similarity_threshold=args.threshold)

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    step = 6400  # 100 ms at 16 kHz int16, like the live stream
    t0 = time.perf_counter()
    n_ok = n_rej = 0
    last_report = t0

    def show(ev):
        nonlocal n_ok, n_rej
        if ev["type"] == "transcript":
            n_ok += 1
            print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f}s match={ev['score']}] {ev['text']}", flush=True)
        elif ev["type"] == "rejected":
            n_rej += 1
            if args.show_rejected:
                print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f}s match={ev['score']}] -- skipped ({ev['reason']})", flush=True)

    for i in range(0, len(pcm), step):
        for ev in pipe.feed_pcm16(pcm[i:i + step]):
            show(ev)
        now = time.perf_counter()
        if now - last_report > 10:  # heartbeat so a long file never looks hung
            last_report = now
            log.info("processed %.0fs of %.0fs audio (%d accepted, %d rejected)",
                     i / 2 / SAMPLE_RATE, audio.size / SAMPLE_RATE, n_ok, n_rej)
    for ev in pipe.flush():
        show(ev)
    log.info("done in %.1fs: %d accepted, %d rejected", time.perf_counter() - t0, n_ok, n_rej)


if __name__ == "__main__":
    main()

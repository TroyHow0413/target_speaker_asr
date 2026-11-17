"""Command-line helpers that use the same engine as the server.

    python scripts/cli.py enroll alice voice_sample.wav
    python scripts/cli.py transcribe alice meeting.wav [--language en] [--threshold 0.55]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tsasr.audio import load_audio_bytes  # noqa: E402
from tsasr.config import load_config  # noqa: E402
from tsasr.engine import Engine  # noqa: E402


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
    engine = Engine(load_config(args.config))
    audio = load_audio_bytes(Path(args.audio).read_bytes())

    if args.cmd == "enroll":
        print(engine.enroll(args.profile, audio, append=args.append))
        return

    pipe = engine.new_session(args.profile, args.language)
    if args.threshold is not None:
        pipe.update(similarity_threshold=args.threshold)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    events = []
    for i in range(0, len(pcm), 6400):  # feed 100 ms at a time, like the live stream
        events += pipe.feed_pcm16(pcm[i:i + 6400])
    events += pipe.flush()
    for ev in events:
        if ev["type"] == "transcript":
            print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f}s match={ev['score']}] {ev['text']}")
        elif ev["type"] == "rejected" and args.show_rejected:
            print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f}s match={ev['score']}] -- skipped ({ev['reason']})")


if __name__ == "__main__":
    main()

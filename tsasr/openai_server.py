# NOTE: deliberately no `from __future__ import annotations` - FastAPI inspects the
# real annotations of the endpoint functions.
"""OpenAI-compatible speech-to-text API that only transcribes the enrolled target speaker.

    POST /v1/audio/transcriptions   (multipart: file, model, language, response_format)
    GET  /v1/models                 (one entry per enrolled profile)

Point any client that supports a custom OpenAI STT endpoint at  http://<host>:8001/v1
`model` selects the voice profile:  "tsasr-alice" or "alice" -> profile `alice`;
anything else (e.g. "whisper-1") -> the default profile (openai.default_profile / --profile).
"""
import argparse
import asyncio
import logging
import re
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .audio import load_audio_bytes
from .base import SAMPLE_RATE
from .config import load_config, resolve_path
from .engine import Engine

log = logging.getLogger("tsasr.openai")

_CJK = re.compile(r"[　-鿿＀-￯]")
_FORMATS = ("json", "text", "verbose_json")


def _join(texts: list) -> str:
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        # no space between CJK text; a space between Latin words
        out = t if not out else out + ("" if _CJK.search(out[-1]) or _CJK.search(t[0]) else " ") + t
    return out


def create_openai_app(cfg: dict, default_profile: Optional[str] = None) -> FastAPI:
    engine = Engine(cfg)
    token = (cfg.get("server") or {}).get("token")
    ocfg = cfg.get("openai") or {}
    fallback_profile = default_profile or ocfg.get("default_profile")
    # one request at a time: models are shared and a tablet can fire several clips quickly
    gate = asyncio.Semaphore(int(ocfg.get("max_concurrent", 1)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def warm():
            try:
                await asyncio.to_thread(engine.warmup)
                log.info("models loaded")
            except Exception:
                log.exception("warm-up failed - check config.yaml")

        task = asyncio.create_task(warm())
        yield
        task.cancel()

    app = FastAPI(title="Target Speaker ASR (OpenAI-compatible)", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def auth(authorization: Optional[str] = Header(None), x_token: Optional[str] = Header(None),
             token_q: Optional[str] = Query(None, alias="token")):
        if not token:
            return
        bearer = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
        if token not in (bearer, x_token, token_q):
            raise HTTPException(401, "bad or missing API key")

    def pick_profile(model: Optional[str]) -> str:
        m = (model or "").strip()
        for cand in (m, m.removeprefix("tsasr-")):
            try:
                if cand and engine.store.exists(cand):
                    return cand
            except ValueError:  # not a valid profile id (e.g. "gpt-4o.transcribe") -> try the default
                pass
        if fallback_profile:
            return fallback_profile
        raise HTTPException(400, f"no voice profile matches model {model!r} and no default_profile is configured. "
                                 f"Use model='tsasr-<profile>' (see GET /v1/models).")

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        data = [{"id": f"tsasr-{p['id'] if isinstance(p, dict) else p}", "object": "model", "owned_by": "tsasr"}
                for p in engine.store.list()]
        return {"object": "list", "data": data}

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
    async def transcribe(file: UploadFile = File(...), model: Optional[str] = Form(None),
                         language: Optional[str] = Form(None), prompt: Optional[str] = Form(None),
                         response_format: str = Form("json"), temperature: Optional[float] = Form(None)):
        if response_format not in _FORMATS:
            raise HTTPException(400, f"response_format must be one of {', '.join(_FORMATS)}")
        profile = pick_profile(model)
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty audio file")

        def work():
            audio = load_audio_bytes(data)
            engine.refresh_profile(profile)
            pipe = engine.new_session(profile, language or None)
            return audio.size / SAMPLE_RATE, pipe.process_audio(audio)

        try:
            async with gate:
                duration, events = await run_in_threadpool(work)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            log.exception("transcription failed")
            raise HTTPException(500, f"{type(e).__name__}: {e}")

        done = [e for e in events if e["type"] == "transcript"]
        text = _join([e["text"] for e in done])
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "json":
            return {"text": text}
        return {
            "task": "transcribe", "language": next((e["language"] for e in done if e.get("language")), language),
            "duration": round(duration, 2), "text": text,
            "segments": [{"id": i, "start": e["t0"], "end": e["t1"], "text": e["text"], "score": e["score"]}
                         for i, e in enumerate(done)],
            "rejected": [{"start": e["t0"], "end": e["t1"], "reason": e["reason"], "score": e["score"]}
                         for e in events if e["type"] == "rejected"],
        }

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Target speaker ASR - OpenAI-compatible API")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None, help="default: openai.port in config.yaml, else 8001")
    ap.add_argument("--profile", default=None, help="default voice profile (overrides openai.default_profile)")
    ap.add_argument("--ssl", action="store_true", help="serve HTTPS using server.ssl_certfile/ssl_keyfile")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    s, o = cfg.get("server") or {}, cfg.get("openai") or {}
    kw = {}
    if args.ssl:
        if not (s.get("ssl_certfile") and s.get("ssl_keyfile")):
            raise SystemExit("--ssl needs server.ssl_certfile and server.ssl_keyfile in config.yaml")
        kw["ssl_certfile"] = resolve_path(s["ssl_certfile"])
        kw["ssl_keyfile"] = resolve_path(s["ssl_keyfile"])
    uvicorn.run(create_openai_app(cfg, args.profile), host=args.host or o.get("host") or s.get("host", "0.0.0.0"),
                port=args.port or int(o.get("port", 8001)), log_level="info", **kw)


if __name__ == "__main__":
    main()

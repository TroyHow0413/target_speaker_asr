# NOTE: deliberately no `from __future__ import annotations` - FastAPI inspects the
# real annotations of the endpoint functions.
import argparse
import asyncio
import json
import logging
import queue
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import load_audio_bytes
from .config import ROOT, load_config, resolve_path
from .custom_models import extract_model_zip
from .engine import Engine
from .openai_server import add_openai_routes

log = logging.getLogger("tsasr")

_FLUSH = object()


class SelectBody(BaseModel):
    kind: str
    preset: Optional[str] = None
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


def create_app(cfg: dict, openai: bool = False, default_profile: Optional[str] = None) -> FastAPI:
    """The web UI + /api + /ws app. With `openai=True` the OpenAI-compatible /v1 routes are served by
    the same app (and so the same loaded models and the same UI settings)."""
    engine = Engine(cfg)
    token = (cfg.get("server") or {}).get("token")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scfg = cfg.get("server") or {}
        idle_s = float(scfg.get("idle_unload_s") or 0)  # 0 = never unload

        async def warm():
            try:
                await asyncio.to_thread(engine.warmup)
                log.info("models loaded")
            except Exception:
                log.exception("warm-up failed - pick another model in the UI or fix config.yaml")

        async def reaper():
            while True:
                await asyncio.sleep(min(30.0, max(5.0, idle_s / 4)))
                try:
                    await asyncio.to_thread(engine.unload_if_idle, idle_s)
                except Exception:
                    log.exception("idle unload failed")

        tasks = []
        if scfg.get("preload", True):
            tasks.append(asyncio.create_task(warm()))
        else:
            log.info("preload off: models load on first use")
        if idle_s > 0:
            log.info("models will be unloaded after %.0fs idle", idle_s)
            tasks.append(asyncio.create_task(reaper()))
        yield
        for t in tasks:
            t.cancel()

    app = FastAPI(title="Target Speaker ASR", lifespan=lifespan)

    def auth(x_token: Optional[str] = Header(None), token_q: Optional[str] = Query(None, alias="token"),
             authorization: Optional[str] = Header(None)):
        # X-Token / ?token= for the web UI, "Authorization: Bearer" for OpenAI-style clients
        bearer = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
        if token and token not in (x_token, token_q, bearer):
            raise HTTPException(401, "bad or missing token")

    # ---- REST ----------------------------------------------------------
    @app.get("/api/state", dependencies=[Depends(auth)])
    async def state():
        return engine.describe()

    @app.post("/api/select", dependencies=[Depends(auth)])
    async def select(body: SelectBody):
        try:
            params = None
            if body.preset is None and body.name is not None:
                # the browser may only choose model + device; everything else comes from config.yaml
                params = engine.client_params(body.kind, body.name, body.params)
            await run_in_threadpool(engine.select, body.kind, body.preset, body.name, params)
        except Exception as e:
            log.exception("select failed")
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        return engine.describe()

    @app.post("/api/pipeline", dependencies=[Depends(auth)])
    async def set_pipeline(body: Dict[str, Any]):
        """Change pipeline defaults (similarity threshold, speaker filter ...) for all new sessions,
        including the OpenAI API, and remember them across restarts."""
        try:
            engine.update_defaults(**body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        return engine.describe()["pipeline"]

    @app.post("/api/enroll", dependencies=[Depends(auth)])
    async def enroll(profile: str = Form(...), append: bool = Form(False),
                     name: Optional[str] = Form(None), file: UploadFile = File(...)):
        data = await file.read()

        def work():
            return engine.enroll(profile, load_audio_bytes(data), append=append, display_name=name)

        try:
            return await run_in_threadpool(work)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            log.exception("enroll failed")
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    @app.delete("/api/profiles/{pid}", dependencies=[Depends(auth)])
    async def delete_profile(pid: str):
        try:
            await run_in_threadpool(engine.delete_profile, pid)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    # ---- user-supplied models -------------------------------------------
    @app.post("/api/models/upload", dependencies=[Depends(auth)])
    async def upload_model(name: str = Form(...), file: UploadFile = File(...)):
        """Upload a Hugging Face model folder zipped (config.json + weights + tokenizer files)."""
        dest_root = Path(resolve_path(cfg.get("custom_models_dir", "models/custom")))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        try:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.close()
            path = await run_in_threadpool(extract_model_zip, tmp.name, dest_root, name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            tmp.close()
            os.unlink(tmp.name)
        try:
            shown = path.relative_to(ROOT).as_posix()
        except ValueError:
            shown = str(path)
        return {"path": shown}

    @app.delete("/api/models/history", dependencies=[Depends(auth)])
    async def forget_model(model: str = Query(...), backend: str = Query("hf_transformers")):
        engine.history.remove(backend, model)  # only forgets the entry; files stay on disk
        return {"ok": True}

    # ---- live audio ------------------------------------------------------
    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket, profile: str, language: Optional[str] = None,
                        token_q: Optional[str] = Query(None, alias="token")):
        await ws.accept()
        if token and token_q != token:
            await ws.send_json({"type": "error", "message": "bad or missing token"})
            await ws.close(code=4401)
            return
        try:
            pipe = await asyncio.to_thread(engine.new_session, profile, language)
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
            await ws.close(code=4400)
            return

        engine.session_opened()
        loop = asyncio.get_running_loop()
        in_q: "queue.Queue" = queue.Queue()
        out_q: "asyncio.Queue" = asyncio.Queue()
        stop = threading.Event()

        def emit(ev):
            loop.call_soon_threadsafe(out_q.put_nowait, ev)

        def worker():  # one thread per session keeps audio order and never blocks the event loop
            while True:
                item = in_q.get()
                if item is None:
                    break
                if stop.is_set():
                    continue
                try:
                    events = pipe.flush() if item is _FLUSH else pipe.feed_pcm16(item)
                    for ev in events:
                        emit(ev)
                except Exception as e:
                    log.exception("pipeline error")
                    emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        async def sender():
            try:
                while True:
                    ev = await out_q.get()
                    if ev is None:
                        return
                    await ws.send_json(ev)
            except Exception:
                stop.set()

        sender_task = asyncio.create_task(sender())
        emit({"type": "ready", "profile": profile})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    in_q.put(msg["bytes"])
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except ValueError:
                        continue
                    if data.get("type") == "config":
                        pipe.update(**{k: v for k, v in data.items() if k != "type"})
                    elif data.get("type") == "flush":
                        in_q.put(_FLUSH)
        except WebSocketDisconnect:
            pass
        finally:
            engine.session_closed()
            stop.set()  # socket is gone: nothing can be delivered, skip queued audio
            in_q.put(None)
            await asyncio.to_thread(thread.join)
            out_q.put_nowait(None)
            await sender_task

    if openai:
        add_openai_routes(app, engine, cfg, auth, default_profile)

    # ---- tablet web client (must be mounted last) -----------------------
    app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    return app


def main(argv=None, openai: bool = False):
    """openai=False: run_server.py  (web UI, HTTPS when certificates are configured, port server.port)
    openai=True:  run_server_openai.py  (web UI + OpenAI-compatible /v1, plain HTTP unless --ssl, port openai.port)"""
    ap = argparse.ArgumentParser(description="Target speaker ASR server" + (" + OpenAI-compatible API" if openai else ""))
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    if openai:
        ap.add_argument("--profile", default=None, help="default voice profile (overrides openai.default_profile)")
        ap.add_argument("--ssl", action="store_true", help="serve HTTPS using server.ssl_certfile/ssl_keyfile")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    s, o = cfg.get("server") or {}, cfg.get("openai") or {}
    have_certs = bool(s.get("ssl_certfile") and s.get("ssl_keyfile"))
    use_ssl = args.ssl if openai else have_certs
    if use_ssl and not have_certs:
        raise SystemExit("--ssl needs server.ssl_certfile and server.ssl_keyfile in config.yaml")
    kw = {}
    if use_ssl:
        kw["ssl_certfile"] = resolve_path(s["ssl_certfile"])
        kw["ssl_keyfile"] = resolve_path(s["ssl_keyfile"])
    port = args.port or int((o.get("port", 8001) if openai else s.get("port", 8000)))
    host = args.host or (o.get("host") if openai else None) or s.get("host", "0.0.0.0")
    app = create_app(cfg, openai=openai, default_profile=getattr(args, "profile", None))
    uvicorn.run(app, host=host, port=port, log_level="info", **kw)


if __name__ == "__main__":
    main()

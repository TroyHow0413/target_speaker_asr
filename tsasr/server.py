# NOTE: deliberately no `from __future__ import annotations` - FastAPI inspects the
# real annotations of the endpoint functions.
import argparse
import asyncio
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import load_audio_bytes
from .config import ROOT, load_config, resolve_path
from .engine import Engine

log = logging.getLogger("tsasr")

_FLUSH = object()


class SelectBody(BaseModel):
    kind: str
    preset: Optional[str] = None
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


def create_app(cfg: dict) -> FastAPI:
    engine = Engine(cfg)
    token = (cfg.get("server") or {}).get("token")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def warm():
            try:
                await asyncio.to_thread(engine.warmup)
                log.info("models loaded")
            except Exception:
                log.exception("warm-up failed - pick another model in the UI or fix config.yaml")

        task = asyncio.create_task(warm())
        yield
        task.cancel()

    app = FastAPI(title="Target Speaker ASR", lifespan=lifespan)

    def auth(x_token: Optional[str] = Header(None), token_q: Optional[str] = Query(None, alias="token")):
        if token and (x_token or token_q) != token:
            raise HTTPException(401, "bad or missing token")

    # ---- REST ----------------------------------------------------------
    @app.get("/api/state", dependencies=[Depends(auth)])
    async def state():
        return engine.describe()

    @app.post("/api/select", dependencies=[Depends(auth)])
    async def select(body: SelectBody):
        try:
            await run_in_threadpool(engine.select, body.kind, body.preset, body.name, body.params)
        except Exception as e:
            log.exception("select failed")
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        return engine.describe()

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
            stop.set()  # socket is gone: nothing can be delivered, skip queued audio
            in_q.put(None)
            await asyncio.to_thread(thread.join)
            out_q.put_nowait(None)
            await sender_task

    # ---- tablet web client (must be mounted last) -----------------------
    app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Target speaker ASR server")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    s = cfg.get("server") or {}
    kw = {}
    if s.get("ssl_certfile") and s.get("ssl_keyfile"):
        kw["ssl_certfile"] = resolve_path(s["ssl_certfile"])
        kw["ssl_keyfile"] = resolve_path(s["ssl_keyfile"])
    uvicorn.run(create_app(cfg), host=args.host or s.get("host", "0.0.0.0"),
                port=args.port or int(s.get("port", 8000)), log_level="info", **kw)


if __name__ == "__main__":
    main()

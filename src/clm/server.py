"""CLM System One API server (FastAPI).

    clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings
    export CLM_API_KEY=...      # optional; then requests need "Authorization: Bearer <key>"

    POST /v1/systemone   {"state": ..., "model": "clm-latest", "questions": {id: Question},
                          "temperature": 1.0}          -> {"model", "answers": {id: Answer}, "usage"}
    POST /v1/rank        {"context": ..., "question": ..., "answers": [...]}
                                                       -> {"model", "ranked": [{rank, candidate, prob}]}
    GET  /v1/models      -> {"models": [{"name", "description", "release_date"}]}
    GET  /health         -> {"ok": true, ...}
    GET  /               -> the playground: a zero-dependency web UI for the endpoint
                            above (``--no-ui`` to leave it off)

Question / Answer objects follow the TypeSafe wire schema (noul / choice /
score), so a request written for TypeSafe replays here unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import shutil
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .embedder import EmbedderError
from .encoders import ENCODERS, IncompatibleEncoder, get_encoder
from .engine import DEFAULT_MODEL, Engine, ModelNotFound, RAW_MODEL
from .heads import default_device

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


_UI_FILES = ("index.html", "app.css", "app.js", "tips.js", "train.js")


def _asset_stamp() -> str:
    """Newest mtime across the UI files, as a short hex tag for their URLs."""
    return format(int(max(os.path.getmtime(os.path.join(STATIC_DIR, f)) for f in _UI_FILES)), "x")


def _index_html() -> str:
    """index.html with ?v=<stamp> on its assets, so a browser that cached an
    older app.css/app.js cannot render the new page with the old styling."""
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    v = _asset_stamp()
    return (html.replace('href="app.css"', f'href="app.css?v={v}"').replace('src="app.js"', f'src="app.js?v={v}"')
            .replace('src="tips.js"', f'src="tips.js?v={v}"').replace('src="train.js"', f'src="train.js?v={v}"'))


class _RevalidatingStatic(StaticFiles):
    """Static files that must be revalidated, so upgrading clm cannot leave a
    stale playground in the browser cache.  The ETag still answers 304."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(engine: Engine, api_key: str | None = None, ui: bool = True, cors: bool = False,
               jobs=None) -> FastAPI:
    """The API, plus the playground at ``/`` unless ``ui=False``.

    ``cors=True`` allows browser requests from any origin (and exposes the
    latency header), which a playground served from somewhere else needs.  It
    is off by default: an API key travels in a header the browser would then
    be free to send from any page.

    ``jobs`` (a ``clm.train_jobs.TrainingJobs``) adds the ``/v1/train/*`` routes the
    playground's Train view uses: upload and inspect a file, start a training job,
    follow its progress, stop it.  Finished heads are served as new models.
    """
    app = FastAPI(title="CLM System One API", version="0.2.0")
    app.state.engine = engine
    if cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                           allow_headers=["*"], allow_credentials=False, expose_headers=["X-CLM-Latency-Ms"])

    def default_model() -> str:
        return getattr(engine, "default_model", DEFAULT_MODEL)

    def auth(authorization: str | None):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")

    @app.get("/health")
    def health():
        out = {"ok": True, "embedder": engine.embedder.healthy(),
               "models": [m["name"] for m in engine.models()], "default_model": default_model(),
               "cache": engine.arena.stats() if engine.arena else None}
        spec = getattr(engine, "encoder", None)
        if spec is not None:            # the playground names the encoder and how to start it
            port = urlsplit(engine.embedder.url).port or 8090
            out["encoder"] = {"name": spec.name, "hf_id": spec.hf_id,
                              "serve": spec.vllm_command(port, getattr(engine.embedder, "max_tokens", None))}
        out["training"] = jobs is not None
        if getattr(engine, "mock", False):      # Claude Code/tools/playground_mock.py: the UI warns about fake numbers
            out["mock"] = True
        return out

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        auth(authorization)
        return {"models": engine.models()}

    def head_of(name: str):
        """The served head behind `name`, or 404 / 422 when there is nothing to export."""
        heads = getattr(engine, "heads", {})
        if name not in heads:
            if name == RAW_MODEL:
                raise HTTPException(422, "clm-raw has no trained head to export")
            raise HTTPException(404, f"unknown model {name!r}; available: {[m['name'] for m in engine.models()]}")
        return heads[name]

    @app.get("/v1/models/{name}/download")
    def model_download(name: str, authorization: str | None = Header(default=None)):
        """The model as one <name>.tar.gz: its head, card and checksums."""
        from starlette.background import BackgroundTask
        from .export import ExportError, archive
        auth(authorization)
        head = head_of(name)
        tmp = tempfile.mkdtemp(prefix="clm-dl-")
        try:
            path = archive(head.path, name, os.path.join(tmp, f"{name}.tar.gz"))
        except ExportError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(422, str(e)) from e
        return FileResponse(path, media_type="application/gzip", filename=f"{name}.tar.gz",
                            background=BackgroundTask(shutil.rmtree, tmp, True))

    @app.post("/v1/models/{name}/export")
    async def model_export(name: str, request: Request, authorization: str | None = Header(default=None)):
        """{dest, endpoint_url?} -> the files written to a folder on this machine or an s3:// bucket."""
        from .export import ExportError, export_model
        auth(authorization)
        head = head_of(name)
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("dest"), str):
            raise HTTPException(422, "body must be {dest: folder or s3://bucket/prefix/}")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, export_model, head.path, body["dest"], name, body.get("endpoint_url") or None)
        except ExportError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"body is not JSON: {e}") from e
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise HTTPException(422, "body must be {state, model, questions}")
        try:
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "temperature must be a number") from e
        t0 = time.perf_counter()
        try:
            out = await asyncio.get_running_loop().run_in_executor(
                None, engine.answer, body["state"], body["questions"], body.get("model") or default_model(), temperature)
        except ModelNotFound as e:
            raise HTTPException(422, str(e.args[0])) from e
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        except EmbedderError as e:
            raise HTTPException(502, str(e)) from e
        return JSONResponse(out, headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})

    @app.post("/v1/rank")
    async def rank(request: Request, authorization: str | None = Header(default=None)):
        """{context, question, answers[, model, temperature]} -> answers ranked best first.

        The plain form of the same primitive: the state head sees ``context + question``,
        the action head sees each answer verbatim.
        """
        auth(authorization)
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"body is not JSON: {e}") from e
        if not isinstance(body, dict) or not isinstance(body.get("answers"), list) or not body["answers"]:
            raise HTTPException(422, "body must be {context, question, answers: [..]}")
        if not all(isinstance(a, str) and a for a in body["answers"]):
            raise HTTPException(422, "answers must be non-empty strings")
        try:
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "temperature must be a number") from e
        t0 = time.perf_counter()
        try:
            ranked = await asyncio.get_running_loop().run_in_executor(
                None, lambda: engine.rank(body.get("context") or "", body["answers"], body.get("question"),
                                          body.get("model") or default_model(), temperature))
        except ModelNotFound as e:
            raise HTTPException(422, str(e.args[0])) from e
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        except EmbedderError as e:
            raise HTTPException(502, str(e)) from e
        return JSONResponse({"model": body.get("model") or default_model(), "ranked": ranked},
                            headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})

    if jobs is not None:
        from .train_jobs import JobError

        def job_call(fn, *a):
            try:
                return fn(*a)
            except JobError as e:
                raise HTTPException(422, str(e)) from e
            except KeyError as e:
                raise HTTPException(404, f"no training job {e.args[0]!r}") from e

        @app.post("/v1/train/inspect")
        async def train_inspect(request: Request, authorization: str | None = Header(default=None)):
            """{filename, content_b64} -> columns, preview and suggested text / label columns."""
            auth(authorization)
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(422, "body must be {filename, content_b64}")
            return await asyncio.get_running_loop().run_in_executor(
                None, job_call, jobs.inspect, body.get("filename") or "", body.get("content_b64") or "")

        @app.get("/v1/train/jobs")
        def train_jobs(authorization: str | None = Header(default=None)):
            auth(authorization)
            return {"jobs": jobs.list()}

        @app.post("/v1/train/jobs")
        async def train_start(request: Request, authorization: str | None = Header(default=None)):
            """Start training: {upload_id, text_column, label_column, question, type, options, ...}."""
            auth(authorization)
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(422, "body must be a JSON object")
            return await asyncio.get_running_loop().run_in_executor(None, job_call, jobs.start, body)

        @app.get("/v1/train/jobs/{job_id}")
        def train_job(job_id: str, authorization: str | None = Header(default=None)):
            auth(authorization)
            return job_call(jobs.get, job_id)

        @app.post("/v1/train/jobs/{job_id}/stop")
        def train_stop(job_id: str, authorization: str | None = Header(default=None)):
            auth(authorization)
            return job_call(jobs.cancel, job_id)

    # mounted last: the routes above shadow it, everything else is the static UI
    if ui and os.path.isdir(STATIC_DIR):
        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def playground():
            return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})

        app.mount("/", _RevalidatingStatic(directory=STATIC_DIR, html=True), name="playground")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("CLM_HOST", "127.0.0.1"),
                    help="interface to listen on (default 127.0.0.1; 0.0.0.0 exposes the API, uploads and "
                         "training to the network, so set CLM_API_KEY too)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLM_PORT", 8700)))
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"))
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default=get_encoder().name,
                    help="frozen encoder behind --emb-url; heads trained on any other encoder are refused "
                         "(default: nemotron-3-embed-8b; CLM_ENCODER overrides)")
    ap.add_argument("--emb-model", default=os.environ.get("CLM_EMB_MODEL"),
                    help="model name the embedder was started with (default: the encoder's name)")
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ["CLM_EMB_MAX_TOKENS"]) if os.environ.get("CLM_EMB_MAX_TOKENS") else None,
                    help="truncate texts to this many tokens before embedding, at most the embedder's "
                         "max-model-len (default: the encoder's budget)")
    ap.add_argument("--ckpt", default=os.environ.get("CLM_CKPT"),
                    help="checkpoint served as clm-latest, trained on --encoder (CLM_CKPT)")
    ap.add_argument("--ckpt-dir", default=None, help="also serve every *.pt in this directory under its file stem")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="serve an extra checkpoint as NAME")
    ap.add_argument("--device", default=None,
                    help="device for the heads (cpu or cuda; default: cuda when available, CLM_DEVICE overrides)")
    ap.add_argument("--action-cache", default=None, metavar="BUDGET",
                    help="GPU memory reserved at start-up for reused state and action vectors: a fraction "
                         "of the device (0.02, the default) or a size (512MiB); 0 disables it. "
                         "Environment: CLM_ACTION_CACHE")
    ap.add_argument("--no-ui", action="store_true", help="do not serve the playground at /")
    ap.add_argument("--runs-dir", default=os.environ.get("CLM_RUNS_DIR",
                                                          os.path.join(os.path.expanduser("~"), ".cache", "clm", "runs")),
                    help="where models trained from the playground are kept (and served again after a restart)")
    ap.add_argument("--no-train", action="store_true", help="turn off training from the playground")
    ap.add_argument("--cors", action="store_true",
                    help="allow browser requests from any origin (needed to drive this server from a "
                         "playground served elsewhere)")
    args = ap.parse_args()

    from .embedder import Embedder
    spec = get_encoder(args.encoder)
    ckpt = args.ckpt
    extra = {}
    for item in args.model:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=PATH, got {item!r}")
        extra[name] = path
    device = args.device or default_device()
    try:
        engine = Engine(Embedder(args.emb_url, args.emb_model, max_tokens=args.max_tokens, encoder=spec),
                        checkpoint=ckpt, models=extra, checkpoint_dir=args.ckpt_dir, device=device,
                        action_cache=args.action_cache)
    except IncompatibleEncoder as e:
        raise SystemExit(f"[clm] {e}") from e
    jobs = None
    if not args.no_train:
        from .train_jobs import TrainingJobs
        jobs = TrainingJobs(engine, args.runs_dir)
        trained = [j["model"] for j in jobs.list() if j.get("served")]
        print(f"[clm] training on; runs in {args.runs_dir}" + (f"; serving trained {trained}" if trained else ""),
              flush=True)
    if not engine.heads:
        print(f"[clm] no head: serving clm-raw only (cosine in {spec.hf_id}'s own space, uncalibrated). "
              "Train one in the playground's Train view, or with clm-train, then serve it with --ckpt.",
              flush=True)
    api_key = os.environ.get("CLM_API_KEY")
    app = create_app(engine, api_key, ui=not args.no_ui, cors=args.cors, jobs=jobs)
    print(f"[clm] models {[m['name'] for m in engine.models()]} on {device}", flush=True)
    print(f"[clm] encoder {spec.name} = {spec.hf_id} ({spec.origin}); {spec.hidden_size}-d, {spec.pooling} pooling, "
          f"{engine.embedder.max_tokens}-token budget, truncation "
          f"{'client-side' if engine.embedder.tokenizer is not None else 'by the embedder'}", flush=True)
    print(f"[clm] embedder {args.emb_url} ({engine.embedder.model}) {'up' if engine.embedder.healthy() else 'NOT REACHABLE'}; "
          f"auth {'on' if api_key else 'off'}", flush=True)
    arena = engine.arena
    if arena:
        pools = " + ".join(f"{p['capacity']:,}x{p['dim']}d" for p in arena.stats()["pools"].values())
        print(f"[clm] vector cache {arena.reserved_mb} MB reserved on {arena.device} ({pools})", flush=True)
    else:
        print("[clm] vector cache off", flush=True)
    print(f"[clm] POST http://{args.host}:{args.port}/v1/systemone", flush=True)
    if not args.no_ui:
        host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
        print(f"[clm] playground http://{host}:{args.port}/", flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

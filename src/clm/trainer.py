"""Training the projection heads on typed questions (``clm-train``, ``finetune.py --task choice``).

The encoder is frozen: every state and option text is embedded once, through the same
prompts and truncation ``clm-serve`` uses, and cached as float16 in
``<out>/embeddings/choice_<encoder>_<max_len>.npz`` (re-runs only embed what is new).
Then only the state head, the action head and the logit scale are trained.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict

import numpy as np

from .embed_backends import Recipe, make_backend
from .encoders import check_compatible, checkpoint_encoder, get_encoder
from .typed_data import typed_decision_examples


class TextCache:
    """sha1(text) -> embedding, persisted as .npz."""

    def __init__(self, path: str):
        self.path, self.vecs = path, {}
        if os.path.exists(path):
            z = np.load(path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
            print(f"[choice] embedding cache: {len(self.vecs)} texts from {path}", flush=True)

    @staticmethod
    def key(t: str) -> str:
        return hashlib.sha1(t.encode()).hexdigest()

    def missing(self, texts):
        return [t for t in dict.fromkeys(texts) if self.key(t) not in self.vecs]

    def update(self, texts, vecs):
        self.vecs.update({self.key(t): np.asarray(v, dtype=np.float16) for t, v in zip(texts, vecs)})

    def add(self, texts, vecs):
        self.update(texts, vecs)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        ks = list(self.vecs)
        np.savez(self.path, keys=np.array(ks), vecs=np.stack([self.vecs[k] for k in ks]).astype(np.float16))

    def __getitem__(self, t):
        return self.vecs[self.key(t)]


EMBED_CHUNK = 256      # texts per encoder call; also how often progress is reported


def train_choice(args, train_rows: list[dict], test_rows: list[dict], source: str = "data",
                 progress=None, embed_fn=None) -> dict:
    """Embed every state and option with the frozen encoder, train the two heads on the
    typed questions in ``train_rows`` (a validation slice is held out by group), and
    write ``best_head.pt`` / ``final_head.pt`` / ``history.json`` to ``args.out_dir``.

    ``args`` carries the options of ``train/finetune.py --task choice`` / ``clm-train``.
    Returns the summary: test accuracy of the head, of ``clm-raw`` and of the majority label.

    ``progress(event: dict)`` is called as work advances (``stage`` = ``embedding`` with
    ``done`` / ``total``, ``baselines``, ``training`` with ``epoch`` / ``epochs`` /
    ``val_acc``, ``testing``); it may raise to cancel the run.  ``embed_fn(texts, role)``
    replaces the encoder backend, e.g. ``clm-serve``'s own embedder; it must return
    L2-normalised vectors for the role-prefixed texts it is given unprefixed.
    """
    report = progress or (lambda event: None)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from clm.engine import RAW_SCALE
    from clm.heads import make_head

    group = lambda r: str(r.get("group") or r["id"])         # noqa: E731  validation holds out whole groups
    ids = sorted({group(r) for r in train_rows})
    random.Random(args.seed).shuffle(ids)
    val_ids = set(ids[:max(1, int(round(args.val_frac * len(ids))))])
    splits = {"train": [r for r in train_rows if group(r) not in val_ids],
              "val": [r for r in train_rows if group(r) in val_ids], "test": test_rows}
    ex = {k: list(typed_decision_examples(v)) for k, v in splits.items()}
    if not ex["train"] or not ex["val"] or not ex["test"]:
        raise SystemExit("need labelled questions in train, validation and test; got " +
                         ", ".join(f"{k} {len(v)}" for k, v in ex.items()))
    print(f"[choice] {source} rows " +
          " ".join(f"{k} {len(splits[k])}" for k in splits) + " | questions " +
          " ".join(f"{k} {len(v)}" for k, v in ex.items()), flush=True)

    spec = get_encoder(args.encoder)
    cache = TextCache(os.path.join(args.embed_cache or os.path.join(args.out_dir, "embeddings"),
                                   f"choice_{spec.name}_{args.max_len}.npz"))
    # states and candidates embed with different prompts, so the cache keys carry the role
    skey, akey = (lambda t: "state\x00" + t), (lambda t: "action\x00" + t)
    examples = [e for v in ex.values() for e in v]
    todo = {"state": cache.missing(skey(e.state_text) for e in examples),
            "action": cache.missing(akey(t) for e in examples for t in e.candidates)}
    total, done = sum(len(v) for v in todo.values()), 0
    report({"stage": "embedding", "done": 0, "total": total})
    if total:
        if embed_fn is None:
            recipe = Recipe(spec, args.max_len, args.embed_model)
            backend = make_backend(args.embed_url, args.embed_model or spec.hf_id, args.max_len,
                                   args.gpu_mem, args.served_model_name, spec.hidden_size)
            embed_fn = lambda texts, role: backend.embed([recipe.ids(t, role) for t in texts])  # noqa: E731
        for role, keys in todo.items():
            if not keys:
                continue
            print(f"[choice] embedding {len(keys)} unique {role} texts with {spec.hf_id}", flush=True)
            for i in range(0, len(keys), EMBED_CHUNK):
                chunk = keys[i:i + EMBED_CHUNK]
                cache.update(chunk, embed_fn([k.split("\x00", 1)[1] for k in chunk], role))
                done += len(chunk)
                report({"stage": "embedding", "done": done, "total": total})
            cache.save()
    hidden = int(next(iter(cache.vecs.values())).shape[-1])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def pack(examples):  # one dense bucket per candidate count
        buckets = defaultdict(list)
        for e in examples:
            buckets[len(e.keys)].append(e)
        out = []
        for k, es in buckets.items():
            q = torch.tensor(np.stack([cache[skey(e.state_text)] for e in es]), dtype=torch.float32)
            c = torch.tensor(np.stack([[cache[akey(t)] for t in e.candidates] for e in es]), dtype=torch.float32)
            out.append((q.to(device), c.to(device),
                        torch.tensor([e.target for e in es], device=device),
                        torch.tensor([e.label for e in es], device=device), [e.qid for e in es]))
        return out

    data = {k: pack(v) for k, v in ex.items()}

    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False) if args.init_ckpt else None
    cfg = dict(width=args.width or 1536, depth=args.depth or 3, activation="gelu", layernorm=True, residual=False)
    if ck:
        c0 = ck.get("cfg", {})
        check_compatible(checkpoint_encoder(c0), spec, what=f"--init-ckpt {args.init_ckpt}")
        cfg.update({k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual") if k in c0})
    proj = (ck or {}).get("cfg", {}).get("projection_dim", args.proj)
    sh = make_head(cfg["width"], cfg["depth"], proj, cfg["activation"], cfg["layernorm"], cfg["residual"],
                   hidden=hidden).to(device)
    ah = make_head(cfg["width"], cfg["depth"], proj, cfg["activation"], cfg["layernorm"], cfg["residual"],
                   hidden=hidden).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    if ck:
        sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float().to(device))
        print(f"[choice] warm start {args.init_ckpt} {cfg}", flush=True)

    def logits(q, c):
        zq = F.normalize(sh(q), dim=-1)
        zc = F.normalize(ah(c.reshape(-1, c.shape[-1])), dim=-1).view(c.shape[0], c.shape[1], -1)
        return logit_scale.exp().clamp(max=100.0) * torch.einsum("bh,bkh->bk", zq, zc)

    def loss_of(lg, tgt, lab):
        if args.targets == "soft":
            return -(tgt * F.log_softmax(lg, -1)).sum(-1).mean()
        return F.cross_entropy(lg, lab)

    if args.loss == "infonce":  # flat train set: every text once, options as indices into it
        tr = ex["train"]
        text_id = {k: i for i, k in enumerate(dict.fromkeys(
            k for e in tr for k in (skey(e.state_text), *map(akey, e.candidates))))}
        emb = torch.tensor(np.stack([cache[k] for k in text_id]), dtype=torch.float32, device=device)
        kmax = max(len(e.keys) for e in tr)
        st_idx = torch.tensor([text_id[skey(e.state_text)] for e in tr], device=device)
        opt_idx = torch.tensor([[text_id[akey(t)] for t in e.candidates] + [-1] * (kmax - len(e.keys)) for e in tr],
                               device=device)
        opt_tgt = torch.tensor([(e.target if args.targets == "soft" else
                                 [float(i == e.label) for i in range(len(e.keys))]) + [0.0] * (kmax - len(e.keys))
                                for e in tr], device=device)

    def infonce_loss(idx):
        """Bidirectional in-batch InfoNCE over the batch's distinct option texts: states -> options
        against the gold distribution, options -> states against the states that hold them."""
        oi, tg = opt_idx[idx], opt_tgt[idx]
        valid = oi >= 0
        pool, col = torch.unique(oi[valid], return_inverse=True)
        cols = torch.zeros_like(oi)
        cols[valid] = col
        tgt = torch.zeros(len(idx), len(pool), device=device).scatter_add_(1, cols, tg * valid)
        zq = F.normalize(sh(emb[st_idx[idx]]), dim=-1)
        zc = F.normalize(ah(emb[pool]), dim=-1)
        lg = logit_scale.exp().clamp(max=100.0) * zq @ zc.t()
        fwd = -(tgt * F.log_softmax(lg, 1)).sum(1).mean()
        w = tgt.t()
        keep = w.sum(1) > 0
        w = w[keep] / w[keep].sum(1, keepdim=True)
        bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
        return (fwd + bwd) / 2

    def raw_logits(q, c):  # clm-serve's clm-raw: cosine in the encoder's own space, no heads
        return RAW_SCALE * torch.einsum("bh,bkh->bk", F.normalize(q, dim=-1), F.normalize(c, dim=-1))

    @torch.no_grad()
    def evaluate(split, raw=False):
        sh.eval(); ah.eval()
        hit = n = 0; ce = 0.0; per = defaultdict(lambda: [0, 0])
        for q, c, tgt, lab, qids in data[split]:
            lg = raw_logits(q, c) if raw else logits(q, c)
            pred = lg.argmax(-1)
            ce += -(tgt * F.log_softmax(lg, -1)).sum(-1).sum().item()
            ok = (pred == lab).tolist()
            hit += sum(ok); n += len(ok)
            for qid, o in zip(qids, ok):
                per[qid][0] += o; per[qid][1] += 1
        sh.train(); ah.train()
        return {"acc": hit / max(1, n), "soft_ce": ce / max(1, n),
                "per_question": {k: round(h / t, 4) for k, (h, t) in sorted(per.items())}}

    def majority_baseline(split):
        counts = defaultdict(lambda: defaultdict(int))
        for e in ex["train"]:
            counts[e.qid][e.label] += 1
        hit = sum(e.label == max(counts[e.qid], key=counts[e.qid].get) for e in ex[split] if counts[e.qid])
        return hit / max(1, len(ex[split]))

    params = list(sh.parameters()) + list(ah.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr or 5e-4, weight_decay=args.weight_decay)
    n_train = sum(len(b[3]) for b in data["train"])
    per_epoch = (math.ceil(n_train / args.batch) if args.loss == "infonce" else
                 sum(math.ceil(len(b[3]) / args.batch) for b in data["train"]))
    steps = max(1, per_epoch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr or 5e-4, total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    raw_val, raw_test = evaluate("val", raw=True), evaluate("test", raw=True)
    base = {"majority_val": majority_baseline("val"), "majority_test": majority_baseline("test"),
            "raw_val": raw_val["acc"], "raw_test": raw_test["acc"],
            "raw_test_per_question": raw_test["per_question"]}
    m0 = {"val": evaluate("val"), "test": evaluate("test")}
    print(f"[choice] baselines: majority test {base['majority_test']:.4f} | clm-raw ({spec.name}, no head) "
          f"val {raw_val['acc']:.4f} test {raw_test['acc']:.4f}", flush=True)
    print(f"[choice] epoch 0 (init) val acc {m0['val']['acc']:.4f} test acc {m0['test']['acc']:.4f}", flush=True)
    report({"stage": "baselines", "majority_test": base["majority_test"], "raw_test": base["raw_test"],
            "raw_val": base["raw_val"]})
    report({"stage": "training", "epoch": 0, "epochs": args.epochs, "val_acc": m0["val"]["acc"]})

    def blob(epoch, metrics):
        return {"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "cfg": {**cfg, "projection_dim": proj, "hidden_size": hidden, "encoder": spec.identity(),
                        "task": "choice", "targets": args.targets, "loss": args.loss, "data": args.data,
                        "workflow": args.workflow, "embed_model": args.embed_model or spec.hf_id,
                        "max_len": args.max_len,
                        "init_ckpt": os.path.basename(args.init_ckpt) if args.init_ckpt else None},
                "epoch": epoch, "metrics": metrics}

    g = torch.Generator().manual_seed(args.seed)
    best, best_ep, bad, history = m0["val"]["acc"], 0, 0, [{"epoch": 0, **m0}]
    torch.save(blob(0, m0["val"]), os.path.join(args.out_dir, "best_head.pt"))
    for ep in range(1, args.epochs + 1):
        tot = nb = 0
        if args.loss == "infonce":  # batches mix candidate counts, so draw from the flat set
            order = [(None, idx) for idx in torch.randperm(len(st_idx), generator=g).split(args.batch)]
        else:
            order = [(bi, idx) for bi, b in enumerate(data["train"])
                     for idx in torch.randperm(len(b[3]), generator=g).split(args.batch)]
            random.Random(args.seed + ep).shuffle(order)
        for bi, idx in order:
            idx = idx.to(device)
            if args.loss == "infonce":
                loss = infonce_loss(idx)
            else:
                q, c, tgt, lab, _ = data["train"][bi]
                loss = loss_of(logits(q[idx], c[idx]), tgt[idx], lab[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        mv = evaluate("val")
        history.append({"epoch": ep, "train_loss": tot / max(1, nb), "val": mv})
        print(f"[choice] epoch {ep} loss {tot/max(1,nb):.4f} val acc {mv['acc']:.4f} "
              f"soft_ce {mv['soft_ce']:.4f}", flush=True)
        report({"stage": "training", "epoch": ep, "epochs": args.epochs, "val_acc": mv["acc"],
                "train_loss": tot / max(1, nb)})
        torch.save(blob(ep, mv), os.path.join(args.out_dir, "final_head.pt"))
        if mv["acc"] > best + 1e-9:
            best, best_ep, bad = mv["acc"], ep, 0
            torch.save(blob(ep, mv), os.path.join(args.out_dir, "best_head.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[choice] early stop at epoch {ep}", flush=True)
                break
    best_ck = torch.load(os.path.join(args.out_dir, "best_head.pt"), map_location="cpu", weights_only=False)
    sh.load_state_dict(best_ck["state_head"]); ah.load_state_dict(best_ck["action_head"])
    with torch.no_grad():
        logit_scale.copy_(best_ck["logit_scale"].to(device))
    report({"stage": "testing"})
    mt = evaluate("test")
    print(f"[choice] best epoch {best_ep}: val acc {best:.4f} | TEST acc {mt['acc']:.4f} "
          f"(clm-raw {base['raw_test']:.4f}, init {m0['test']['acc']:.4f}, majority {base['majority_test']:.4f}) "
          f"per_question {mt['per_question']}",
          flush=True)
    json.dump(history, open(os.path.join(args.out_dir, "history.json"), "w"), indent=1)
    return {"best_epoch": best_ep, "val_acc": best, "test": mt, "init_test": m0["test"], **base,
            "train_questions": n_train}

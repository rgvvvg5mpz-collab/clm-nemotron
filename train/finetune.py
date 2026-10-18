#!/usr/bin/env python3
"""Fine-tune the CLM projection heads on a frozen encoder (default Nemotron-3-Embed-8B).

Tasks (``--task``)
  clm     (state, action) step traces from DeepSWE or custom data;
          one shared in-batch group-masked InfoNCE trainer.
          ``--holdout-folds`` trains one head per fold and writes fold_spec.json
          for evaluation/bon_eval.py. ``--folds K --fold-index INDEX`` creates
          task-disjoint folds stratified by candidate/pass count before training.
  choice  typed System One questions (LocalLLaMA/typed-decisions), candidates built by
          clm.schema.build_pairs. ``--loss infonce`` (default): bidirectional in-batch InfoNCE
          over the batch's distinct option texts; ``softce``: softmax over each question's own
          candidates. Both train against the annotator distribution (``--targets soft``) or the
          gold label (``hard``); evaluation is always per question.

Embeddings (clm)
  --emb-dir DIR        embedding dir (embed_shard.py + merge_embeddings.py output)
  --hf-dataset REPO    native or parquet embedding dataset written by this fork
  --data FILE          step transitions (.json / .jsonl), embedded from scratch
choice always embeds from scratch, with offline vLLM or a vLLM pooling server
(``--embed-url``).

Encoder (``--encoder``, default nemotron-3-embed-8b)
  Heads are tied to the frozen encoder they were trained on: its identity is recorded in
  every embedding dir and checkpoint (``cfg["encoder"]``), and an embedding dir, warm-start
  checkpoint or dataset from another encoder, or one that records no encoder (every
  upstream CLM release), is refused.  Train from text: ``--data`` for clm,
  typed-decisions for choice.

Checkpoints hold state_head / action_head / logit_scale / cfg.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "preprocessing"))
sys.path.insert(0, HERE)
import adapters  # noqa: E402
import embed_utils  # noqa: E402
import hf_embeddings  # noqa: E402
from clm.encoders import ENCODERS, check_compatible, checkpoint_encoder, get_encoder  # noqa: E402
from clm.heads import make_head  # noqa: E402
from clm.trainer import train_choice  # noqa: E402
from clm.typed_data import load_typed_rows  # noqa: E402


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


# --------------------------------------------------------------------------- #
# clm
# --------------------------------------------------------------------------- #
def _make_stratified_folds(index_path: str, k: int, out_dir: str,
                           config: str | None = None) -> tuple[list[str], dict]:
    """Write task-disjoint folds balanced by candidate count and pass count."""
    data = json.load(open(index_path))
    records = data if isinstance(data, list) else data.get("rows", data.get("records"))
    if not isinstance(records, list):
        raise ValueError(f"{index_path}: expected a list or an object containing rows/records")

    by_task = collections.defaultdict(list)
    for record in records:
        row_config = (record.get("config") or record.get("job_id") or
                      record.get("effort") or "default")
        if config is not None and row_config != config:
            continue
        task = record.get("task_name") or record.get("task_id")
        reward = record.get("passed")
        if reward is None:
            reward = record.get("reward")
        if reward is None:
            reward = (record.get("rewards") or {}).get("reward")
        if task is not None and reward is not None:
            by_task[task].append(float(reward) > 0.5)
    if len(by_task) < k:
        suffix = f" for config {config!r}" if config is not None else ""
        raise ValueError(f"{index_path}: need at least {k} indexed tasks{suffix}; found {len(by_task)}")

    strata = collections.defaultdict(list)
    for task, outcomes in by_task.items():
        strata[(len(outcomes), sum(outcomes))].append(task)
    folds = [[] for _ in range(k)]
    position = 0
    for key in sorted(strata):
        for task in sorted(strata[key]):
            folds[position % k].append(task)
            position += 1

    fold_dir = os.path.join(out_dir, "folds")
    os.makedirs(fold_dir, exist_ok=True)
    paths, summary = [], {"index": os.path.abspath(index_path), "config": config, "k": k,
                          "stratification": ["candidate_count", "pass_count"], "folds": []}
    for number, tasks in enumerate(folds):
        path = os.path.join(fold_dir, f"fold{number}.json")
        tasks = sorted(tasks)
        json.dump(tasks, open(path, "w"), indent=1)
        pass_counts = collections.Counter(sum(by_task[task]) for task in tasks)
        mixed = sum(0 < sum(by_task[task]) < len(by_task[task]) for task in tasks)
        summary["folds"].append({"path": os.path.relpath(path, out_dir), "tasks": len(tasks),
                                 "mixed_tasks": mixed,
                                 "pass_count_histogram": dict(sorted(pass_counts.items()))})
        paths.append(path)
        print(f"[folds] fold{number}: {len(tasks)} tasks ({mixed} mixed/decidable)", flush=True)
    json.dump(summary, open(os.path.join(fold_dir, "manifest.json"), "w"), indent=1)
    print(f"[folds] {len(by_task)} tasks -> {fold_dir}", flush=True)
    return paths, summary


def clm_embedding_dir(args) -> str:
    """Resolve the CLM data source to an embedding dir (building it if needed)."""
    if args.emb_dir:
        if not hf_embeddings.is_embedding_dir(args.emb_dir):
            raise SystemExit(f"--emb-dir {args.emb_dir} is not an embedding dir")
        return args.emb_dir
    cache = args.embed_cache or os.path.join(args.out_dir, "embeddings")
    if args.hf_dataset:
        return hf_embeddings.download(args.hf_dataset, os.path.join(cache, _slug(args.hf_dataset)),
                                      cache_dir=args.hf_cache, split=args.hf_split)
    spec = get_encoder(args.encoder)
    out = os.path.join(cache, f"clm_{spec.name}_{args.max_len}_" +
                       _slug(os.path.splitext(os.path.basename(args.data))[0]))
    if hf_embeddings.is_embedding_dir(out):
        print(f"[clm] reusing embeddings in {out}", flush=True)
        return out
    recs = adapters.read_transitions(args.data)
    recipe = embed_utils.Recipe(spec, args.max_len, args.embed_model)
    backend = embed_utils.make_backend(args.embed_url, args.embed_model or spec.hf_id, args.max_len, args.gpu_mem,
                                       args.served_model_name, spec.hidden_size)
    print(f"[clm] embedding {len(recs)} steps from scratch with {spec.hf_id}", flush=True)
    se = backend.embed([recipe.ids(r["state"], "state") for r in recs])
    ae = backend.embed([recipe.ids(r["action"], "action") for r in recs])
    os.makedirs(out, exist_ok=True)
    torch.save(torch.from_numpy(se.astype(np.float16)), os.path.join(out, "state_embeddings.pt"))
    torch.save(torch.from_numpy(ae.astype(np.float16)), os.path.join(out, "action_embeddings.pt"))
    meta = {"num_samples": len(recs), "hidden_size": int(se.shape[1]),
            "model_name": args.embed_model or spec.hf_id, "encoder": spec.identity(),
            "max_model_len": args.max_len, "input_file": os.path.abspath(args.data),
            "samples": [{k: r.get(k) for k in hf_embeddings.META_KEYS} for r in recs]}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"))
    return out


def _load_clm_data(path, device, keep_tasks=None):
    """Load one embedding directory, optionally restricted to a task set."""
    states = torch.load(os.path.join(path, "state_embeddings.pt"), map_location="cpu")
    actions = torch.load(os.path.join(path, "action_embeddings.pt"), map_location="cpu")
    samples = json.load(open(os.path.join(path, "metadata.json")))["samples"]
    if not (len(samples) == len(states) == len(actions)):
        raise ValueError(f"{path}: embedding and metadata lengths differ")
    if keep_tasks is not None:
        keep = torch.tensor([i for i, sample in enumerate(samples)
                             if sample["task_id"] in keep_tasks], dtype=torch.long)
        states, actions = states[keep], actions[keep]
        samples = [samples[i] for i in keep.tolist()]
    if len(samples) < 2:
        raise ValueError(f"{path}: need at least two samples after task filtering")
    tasks = sorted({sample["task_id"] for sample in samples})
    task_ids = {task: i for i, task in enumerate(tasks)}
    task_idx = torch.tensor([task_ids[sample["task_id"]] for sample in samples])
    groups = {}
    codes = torch.tensor([groups.setdefault((sample["task_id"], sample["step_idx"]), len(groups))
                          for sample in samples])
    return states.to(device), actions.to(device), codes.to(device), task_idx, samples, len(tasks)


def _clm_loss(state_head, action_head, logit_scale, states, actions, codes):
    state_z = F.normalize(state_head(states), dim=-1)
    action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * state_z @ action_z.t()
    labels = torch.arange(len(states), device=states.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    forward = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    backward = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (forward + backward) / 2, logits, labels


@torch.no_grad()
def _evaluate_clm(state_head, action_head, logit_scale, data, batch):
    states, actions, codes, task_idx, _, n_tasks = data
    state_head.eval(); action_head.eval()
    n = len(states)
    losses, top1, seen = [], 0, 0
    for start in range(0, n, min(batch, n)):
        stop = min(start + batch, n)
        if stop - start < 2:
            continue
        loss, logits, labels = _clm_loss(
            state_head, action_head, logit_scale, states[start:stop].float(),
            actions[start:stop].float(), codes[start:stop])
        losses.append(loss.item())
        top1 += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)

    state_z, action_z = [], []
    for start in range(0, n, 8192):
        state_z.append(F.normalize(state_head(states[start:start + 8192].float()), dim=-1))
        action_z.append(F.normalize(action_head(actions[start:start + 8192].float()), dim=-1))
    state_z, action_z = torch.cat(state_z), torch.cat(action_z)
    within_hits = within_n = 0
    within_ranks = []
    for task in range(n_tasks):
        members = (task_idx == task).nonzero(as_tuple=True)[0][:4096]
        if len(members) < 2:
            continue
        device_members = members.to(states.device)
        similarity = state_z[device_members] @ action_z[device_members].t()
        member_codes = codes[device_members]
        same = member_codes.unsqueeze(0) == member_codes.unsqueeze(1)
        same.fill_diagonal_(False)
        similarity = similarity.masked_fill(same, float("-inf"))
        rank = (similarity > similarity.diagonal().unsqueeze(1)).sum(1)
        within_hits += (rank == 0).sum().item()
        within_n += len(members)
        within_ranks.append((rank.float() / (len(members) - 1)).mean().item())
    state_head.train(); action_head.train()
    return {"val_loss": sum(losses) / max(1, len(losses)),
            "val_top1": top1 / max(1, seen),
            "within_task_top1": within_hits / max(1, within_n),
            "within_task_meanrank": sum(within_ranks) / max(1, len(within_ranks)),
            "n_val": n}


def _clm_batches(n, batch, sampler, task_idx, generator, tasks_per_batch):
    def chunks(indices):
        return [indices[i:i + batch] for i in range(0, len(indices), batch)
                if len(indices[i:i + batch]) >= 2]

    if sampler == "random":
        return chunks(torch.randperm(n, generator=generator))
    batches, pending = [], []
    unique = task_idx.unique().tolist()
    for position in torch.randperm(len(unique), generator=generator).tolist():
        indices = (task_idx == unique[position]).nonzero(as_tuple=True)[0]
        pending.append(indices[torch.randperm(len(indices), generator=generator)])
        if len(pending) == tasks_per_batch:
            block = torch.cat(pending); pending = []
            batches.extend(chunks(block[torch.randperm(len(block), generator=generator)]))
    if pending:
        block = torch.cat(pending)
        batches.extend(chunks(block[torch.randperm(len(block), generator=generator)]))
    return batches


def _train_clm(args, emb_dir, out_dir, holdout_path):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    checkpoint = (torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
                  if args.init_ckpt else None)
    initial = (checkpoint or {}).get("cfg", {})
    width = args.width or initial.get("width", 1536)
    depth = args.depth or initial.get("depth", 3)
    head_options = dict(activation=initial.get("activation", "gelu"),
                        layernorm=initial.get("layernorm", True),
                        residual=initial.get("residual", False))

    metadata = json.load(open(os.path.join(emb_dir, "metadata.json")))
    encoder = checkpoint_encoder(metadata)
    check_compatible(encoder, get_encoder(args.encoder), what=f"embedding dir {emb_dir}")
    if checkpoint:
        check_compatible(checkpoint_encoder(initial), encoder, what=f"--init-ckpt {args.init_ckpt}")
    pool_tasks = sorted({sample["task_id"] for sample in metadata["samples"]})
    holdout = set(json.load(open(holdout_path))) if holdout_path else set()
    usable = [task for task in pool_tasks if task not in holdout]
    if len(usable) < 2:
        raise ValueError("CLM training needs at least two non-held-out tasks")
    shuffled = list(usable); random.Random(args.seed).shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(args.val_frac * len(shuffled))))
    val_tasks, train_tasks = set(shuffled[:n_val]), set(shuffled[n_val:])
    train = _load_clm_data(emb_dir, device, train_tasks)
    val = _load_clm_data(emb_dir, device, val_tasks)
    hidden = int(train[0].shape[1])
    if val[0].shape[1] != hidden:
        raise ValueError("training and validation embedding widths differ")
    if checkpoint and initial.get("hidden_size", hidden) != hidden:
        raise ValueError("checkpoint and dataset embedding widths differ")
    json.dump({"holdout": sorted(holdout), "train_tasks": sorted(train_tasks),
               "val_tasks": sorted(val_tasks)}, open(os.path.join(out_dir, "task_split.json"), "w"))
    print(f"[clm] tasks: pool {len(pool_tasks)}, holdout {len(holdout)}, "
          f"train {len(train_tasks)}, val {len(val_tasks)}", flush=True)
    print(f"[clm] train {len(train[0])} pairs | val {len(val[0])} pairs", flush=True)

    state_head = make_head(width, depth, args.proj, hidden=hidden, **head_options).to(device)
    action_head = make_head(width, depth, args.proj, hidden=hidden, **head_options).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    if checkpoint:
        state_head.load_state_dict(checkpoint["state_head"])
        action_head.load_state_dict(checkpoint["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(checkpoint["logit_scale"]).to(device))

    params = list(state_head.parameters()) + list(action_head.parameters()) + [logit_scale]
    lr = args.lr or 2e-3 * math.sqrt(1024 / width) * math.sqrt(args.batch / 1024)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    initial_batches = _clm_batches(len(train[0]), args.batch, args.sampler, train[3],
                                   torch.Generator().manual_seed(0), args.tasks_per_batch)
    if not initial_batches:
        raise ValueError("CLM training produced no batches; lower --batch or change --sampler")
    total_steps = len(initial_batches) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos")

    def metric_value(metrics):
        return (-metrics["val_loss"] if args.clm_select_metric == "val_loss"
                else metrics["within_task_top1"])

    def checkpoint_blob(epoch, metrics):
        return {"state_head": state_head.state_dict(), "action_head": action_head.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "cfg": {"width": width, "depth": depth, "projection_dim": args.proj,
                        "hidden_size": hidden, "encoder": encoder, **head_options, "init_ckpt": args.init_ckpt,
                        "sampler": args.sampler, "batch": args.batch, "lr": lr,
                        "train_dir": emb_dir}, "epoch": epoch, "metrics": metrics}

    initial_metrics = _evaluate_clm(state_head, action_head, logit_scale, val, args.batch)
    history = [{"epoch": 0, **initial_metrics}]
    best, best_epoch, stale = metric_value(initial_metrics), 0, 0
    torch.save(checkpoint_blob(0, initial_metrics), os.path.join(out_dir, "best_head.pt"))
    print(f"[clm] epoch 0 {json.dumps(initial_metrics)}", flush=True)
    started = time.time()
    scheduler_steps = 0
    for epoch in range(1, args.epochs + 1):
        batches = _clm_batches(len(train[0]), args.batch, args.sampler, train[3],
                               generator, args.tasks_per_batch)
        train_loss = 0.0
        for indices in batches:
            indices = indices.to(device)
            loss, _, _ = _clm_loss(state_head, action_head, logit_scale,
                                    train[0][indices].float(), train[1][indices].float(),
                                    train[2][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if scheduler_steps < total_steps:
                scheduler.step()
                scheduler_steps += 1
            train_loss += loss.item()
        metrics = _evaluate_clm(state_head, action_head, logit_scale, val, args.batch)
        metrics.update(train_loss=train_loss / len(batches), epoch=epoch,
                       minutes=(time.time() - started) / 60, logit_scale=logit_scale.item())
        history.append(metrics)
        print(f"[clm] epoch {epoch} {json.dumps(metrics)}", flush=True)
        torch.save(checkpoint_blob(epoch, metrics), os.path.join(out_dir, "final_head.pt"))
        score = metric_value(metrics)
        if score > best + 1e-4:
            best, best_epoch, stale = score, epoch, 0
            torch.save(checkpoint_blob(epoch, metrics), os.path.join(out_dir, "best_head.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[clm] early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break
    summary = {"history": history, "best_epoch": best_epoch,
               "select_metric": args.clm_select_metric, "best_score": best}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary


def run_clm(args) -> dict:
    emb_dir = clm_embedding_dir(args)
    fold_summary = None
    if args.folds:
        args.holdout_folds, fold_summary = _make_stratified_folds(
            args.fold_index, args.folds, args.out_dir, args.fold_config)
        pool_tasks = {sample["task_id"] for sample in
                      json.load(open(os.path.join(emb_dir, "metadata.json")))["samples"]}
        fold_tasks = set().union(*(set(json.load(open(path))) for path in args.holdout_folds))
        missing = sorted(fold_tasks - pool_tasks)
        if missing:
            preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise ValueError(
                f"{args.folds}-fold cross-validation needs training embeddings for every indexed task; "
                f"{len(missing)}/{len(fold_tasks)} are absent ({preview}). Supply the complete "
                "embedding store instead of a task subset.")
    runs = ([(f"fold{k}", f) for k, f in enumerate(args.holdout_folds)] if args.holdout_folds
            else [(".", args.holdout_tasks)])
    spec, results = [], {}
    for name, holdout in runs:
        out = os.path.normpath(os.path.join(args.out_dir, name))
        os.makedirs(out, exist_ok=True)
        print(f"[clm] training {name} -> {out}", flush=True)
        summ = _train_clm(args, emb_dir, out, holdout)
        results[name] = {"best_epoch": summ["best_epoch"], summ["select_metric"]: summ["best_score"]}
        if args.holdout_folds:
            spec.append({"checkpoint": f"{name}/best_head.pt", "tasks": sorted(json.load(open(holdout)))})
    if spec:
        path = os.path.join(args.out_dir, "fold_spec.json")
        json.dump(spec, open(path, "w"), indent=1)
        print(f"[clm] wrote {path}", flush=True)
    return {"embeddings": emb_dir, "folds": fold_summary, "runs": results}


# --------------------------------------------------------------------------- #
# choice
# --------------------------------------------------------------------------- #
def run_choice(args) -> dict:
    train_rows = load_typed_rows(args.data, "train", args.workflow, args.hf_cache)
    test_rows = load_typed_rows(args.data, "test", args.workflow, args.hf_cache)
    return train_choice(args, train_rows, test_rows, source=f"{args.data} [{args.workflow}]")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["clm", "choice"], required=True)
    ap.add_argument("--out-dir", required=True)
    src = ap.add_argument_group("data")
    src.add_argument("--emb-dir", help="clm: embedding dir to train on")
    src.add_argument("--hf-dataset",
                     help="clm: native/parquet embedding dataset written by this fork (HF id or local path)")
    src.add_argument("--data", help="clm: transitions .json/.jsonl (embedded from scratch); "
                                    "choice: typed-decisions HF id, local dir or parquet file")
    src.add_argument("--workflow", default="all", help="choice: typed-decisions config")
    src.add_argument("--hf-cache", default=None, help="Hugging Face download cache")
    src.add_argument("--hf-split", default=None, help="native HF embedding subdirectory, e.g. train")
    emb = ap.add_argument_group("from-scratch embedding")
    emb.add_argument("--encoder", choices=sorted(ENCODERS), default=get_encoder().name,
                     help="frozen encoder the heads are trained on (default: nemotron-3-embed-8b; "
                          "CLM_ENCODER overrides); prompts and truncation follow clm.encoders")
    emb.add_argument("--embed-model", default=None,
                     help="local path / HF id of the encoder weights and tokenizer (default: the encoder's)")
    emb.add_argument("--embed-url", default=None, help="vLLM /v1/embeddings server; offline vLLM if unset")
    emb.add_argument("--served-model-name", default=None, help="model name the server was started with")
    emb.add_argument("--max-len", type=int, default=None, help="token budget (default 8192 clm, 2048 choice)")
    emb.add_argument("--gpu-mem", type=float, default=0.85)
    emb.add_argument("--embed-cache", default=None, help="where embeddings are cached (default OUT/embeddings)")
    hd = ap.add_argument_group("heads / optimisation")
    hd.add_argument("--init-ckpt", default=None, help="warm-start checkpoint")
    hd.add_argument("--width", type=int, default=None)
    hd.add_argument("--depth", type=int, default=None)
    hd.add_argument("--proj", type=int, default=512)
    hd.add_argument("--epochs", type=int, default=20)
    hd.add_argument("--batch", type=int, default=None, help="default 2048 clm, 256 choice")
    hd.add_argument("--lr", type=float, default=None, help="default: CLM width/batch rule, 5e-4 choice")
    hd.add_argument("--weight-decay", type=float, default=0.0)
    hd.add_argument("--val-frac", type=float, default=0.1)
    hd.add_argument("--patience", type=int, default=5)
    hd.add_argument("--seed", type=int, default=1234)
    hd.add_argument("--gpu", type=int, default=0)
    clm = ap.add_argument_group("clm only")
    clm.add_argument("--holdout-folds", nargs="+", default=None, metavar="FOLD_JSON",
                    help="train one head per fold (each holds out that fold's tasks) + write fold_spec.json")
    clm.add_argument("--holdout-tasks", default=None, help="single run holding out these tasks")
    clm.add_argument("--folds", type=int, default=None, metavar="K",
                    help="generate K stratified task folds from --fold-index and train them")
    clm.add_argument("--fold-index", default=None,
                    help="trial index used by --folds; strata are candidate count and pass count")
    clm.add_argument("--fold-config", default=None,
                    help="optional config/job/effort value selected from --fold-index")
    clm.add_argument("--sampler", choices=["random", "task-blocked"], default="random")
    clm.add_argument("--tasks-per-batch", type=int, default=4)
    clm.add_argument("--clm-select-metric", choices=["within_task_top1", "val_loss"],
                     default="within_task_top1")
    ch = ap.add_argument_group("choice only")
    ch.add_argument("--targets", choices=["soft", "hard"], default="soft",
                    help="train on annotator distributions (soft) or gold labels (hard)")
    ch.add_argument("--loss", choices=["infonce", "softce"], default="infonce",
                    help="bidirectional in-batch InfoNCE over the batch's distinct option texts (infonce) "
                         "or a softmax over each question's own options (softce)")
    args = ap.parse_args()

    if args.epochs < 1 or args.patience < 1:
        ap.error("--epochs and --patience must be positive")
    if args.batch is not None and args.batch < 2:
        ap.error("--batch must be at least 2")

    if args.task == "clm":
        sources = sum(bool(x) for x in (args.emb_dir, args.hf_dataset, args.data))
        if sources != 1:
            ap.error("clm needs exactly one of --data (transitions .jsonl, embedded with --encoder), "
                     "--emb-dir or --hf-dataset")
        if sum(bool(x) for x in (args.holdout_folds, args.holdout_tasks, args.folds)) > 1:
            ap.error("--holdout-folds, --holdout-tasks, and --folds are mutually exclusive")
        if args.folds is not None and args.folds < 2:
            ap.error("--folds must be at least 2")
        if bool(args.folds) != bool(args.fold_index):
            ap.error("--folds and --fold-index must be supplied together")
        if args.fold_config and not args.folds:
            ap.error("--fold-config requires --folds")
        args.max_len = args.max_len or 8192
        args.batch = args.batch or 2048
    else:
        if args.hf_split:
            ap.error("--hf-split is a clm option")
        if not args.data:
            ap.error("choice needs --data (e.g. LocalLLaMA/typed-decisions)")
        if (args.emb_dir or args.hf_dataset or args.holdout_folds or args.holdout_tasks or
                args.folds or args.fold_index or args.fold_config):
            ap.error("--emb-dir/--hf-dataset/--holdout-*/--fold-* are clm options")
        args.max_len = args.max_len or 2048
        args.batch = args.batch or 256
    for p in (args.init_ckpt, *(args.holdout_folds or []), args.holdout_tasks, args.fold_index):
        if p and not os.path.exists(p):
            ap.error(f"file not found: {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    res = run_clm(args) if args.task == "clm" else run_choice(args)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    json.dump({**res, "args": vars(args)}, open(os.path.join(args.out_dir, "finetune_summary.json"), "w"),
              indent=1, default=str)
    print(f"[done] {json.dumps(res, default=str)}", flush=True)


if __name__ == "__main__":
    main()

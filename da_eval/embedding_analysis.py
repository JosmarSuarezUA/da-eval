"""
da_eval.embedding_analysis
==========================
``da-eval embeddings`` command: analyse the image embeddings saved by the
evaluation pipeline for one source model, with ``EmbeddingEvaluator``.

- Projections (PCA / t-SNE / UMAP / PaCMAP) of all groups on a balanced subset,
  saved as PNG and logged to W&B as ``embeddings/<method>``.
- pymdma distribution metrics (improved precision/recall, density, coverage)
  with the **source train** embeddings as reference, against the source test
  and every target test split (full sets).

Input is a source results folder (``results/<model>/source_<label>``) or the
W&B artifact the evaluation logged (``embeddings-<model>-<source>``)::

    uv run da-eval embeddings --results results/rtdetr/source_AFO --wandb-project rtdetrv4
    uv run da-eval embeddings --artifact embeddings-rtdetr-C:latest --wandb-project rtdetrv4

Needs the analysis extra: ``pip install "da-eval[analysis]"``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("pca", "tsne", "umap", "pacmap")
PYMDMA_METRICS = ("improved_precision", "improved_recall", "density", "coverage")


# ---------------------------------------------------------------------------
# 1. Loading the embedding groups of one source
# ---------------------------------------------------------------------------

def load_manifest(folder: str | Path) -> dict:
    """Read ``embeddings_manifest.json``, or rebuild it from file names for older results.

    Older results (before manifests existed) only have ``embeddings_<S>_train.npy``
    and ``embeddings_<S>_to_<T>.npy``; groups are then named after the dataset keys.
    """
    folder = Path(folder)
    path = folder / "embeddings_manifest.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)

    groups = []
    sources = set()
    for f in sorted(folder.glob("embeddings_*_to_*.npy")):
        m = re.fullmatch(r"embeddings_(.+)_to_(.+)\.npy", f.name)
        sources.add(m.group(1))
        groups.append({"dataset": m.group(2), "split": "test", "embeddings": f.name})
    if len(sources) != 1:
        raise FileNotFoundError(
            f"{folder}: no embeddings_manifest.json and could not infer a single source from "
            f"embeddings_<source>_to_<target>.npy files (found sources: {sorted(sources)})"
        )
    source = sources.pop()
    for g in groups:
        g["role"] = "source" if g["dataset"] == source else "target"
    train = folder / f"embeddings_{source}_train.npy"
    if train.exists():
        groups.insert(0, {"dataset": source, "split": "train", "role": "source", "embeddings": train.name})
    for g in groups:
        g["label"] = g["dataset"]
        g["name"] = f"{g['dataset']} {g['split']}"
        images = folder / g["embeddings"].replace(".npy", "_images.csv")
        g["images"] = images.name if images.exists() else None
    return {"source": source, "source_label": source, "groups": groups}


def ordered_groups(manifest: dict) -> list[dict]:
    """Source train first, then source test, then the targets in manifest order."""
    rank = {("source", "train"): 0, ("source", "test"): 1}
    return sorted(manifest["groups"], key=lambda g: rank.get((g["role"], g["split"]), 2))


# ---------------------------------------------------------------------------
# 2. Analysis
# ---------------------------------------------------------------------------

def plot_projections(
    evaluator: Any,
    methods: list[str],
    title_prefix: str,
    out_dir: Path,
    balanced: bool,
    seed: int,
    run: Any = None,
) -> dict[str, Path]:
    """Compute and plot each projection; save PNGs and optionally log them to W&B."""
    import matplotlib.pyplot as plt

    n = len(evaluator._get_data(balanced)[0])
    runners = {
        "pca": lambda: evaluator.run_pca(balanced=balanced),
        "tsne": lambda: evaluator.run_tsne(balanced=balanced, random_state=seed, perplexity=min(30, n - 1)),
        "umap": lambda: evaluator.run_umap(balanced=balanced, random_state=seed),
        "pacmap": lambda: evaluator.run_pacmap(balanced=balanced, random_state=seed),
    }
    names = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP", "pacmap": "PaCMAP"}
    saved = {}
    for method in methods:
        coords = runners[method]()
        fig = evaluator.plot_embeddings(coords, title=f"{title_prefix} ({names[method]})", balanced=balanced)
        path = out_dir / f"{method}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if run is not None:
            import wandb
            run.log({f"embeddings/{method}": wandb.Image(fig)})
        plt.close(fig)
        saved[method] = path
        print(f"[*] {names[method]} plot -> {path}")
    return saved


def _read_images(folder: Path, group: dict) -> list[dict] | None:
    if not group.get("images"):
        return None
    with open(folder / group["images"]) as fh:
        return list(csv.DictReader(fh))


def compute_group_metrics(
    evaluator: Any,
    groups: list[dict],
    folder: Path,
    out_dir: Path,
    k: int,
    metric: str,
    n_workers: int,
) -> list[dict]:
    """pymdma metrics of each group against the source-train reference (full sets).

    Returns one row per compared group; per-image scores are written to
    ``instance_scores_<dataset>_<split>.csv`` (joined with image ids when available).
    """
    reference = next((g for g in groups if g["role"] == "source" and g["split"] == "train"), None)
    if reference is None:
        print("[!] No source train embeddings: pymdma metrics need them as reference; skipped.")
        return []
    ref_emb = evaluator.embeddings_dict[reference["name"]]

    rows = []
    for g in groups:
        if g is reference:
            continue
        emb = evaluator.embeddings_dict[g["name"]]
        row = {
            "dataset": g["dataset"], "label": g.get("label", g["dataset"]), "split": g["split"], "role": g["role"],
            "reference": reference["name"], "n_reference": len(ref_emb), "n_images": len(emb), "k": k, "metric": metric,
        }
        try:
            result = evaluator.compute_distribution_metrics(ref_emb, emb, k=k, metric=metric, n_workers=n_workers)
        except ValueError as exc:  # e.g. fewer than k+1 images
            print(f"[!] {g['name']}: pymdma metrics skipped ({exc})")
            rows.append({**row, **{m: None for m in PYMDMA_METRICS}})
            continue
        row.update({m: float(np.asarray(result[m]["dataset"]).mean()) for m in PYMDMA_METRICS})
        rows.append(row)

        images = _read_images(folder, g)
        instance_path = out_dir / f"instance_scores_{g['dataset']}_{g['split']}.csv"
        with open(instance_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            per_image = {m: np.asarray(result[m]["instance"]).ravel() for m in PYMDMA_METRICS}
            usable = [m for m in PYMDMA_METRICS if len(per_image[m]) == len(emb)]
            writer.writerow(["row", "image_id", "file_name", *usable])
            for i in range(len(emb)):
                img = images[i] if images else {}
                writer.writerow([i, img.get("image_id", ""), img.get("file_name", ""), *(per_image[m][i] for m in usable)])
        print(f"[*] {g['name']}: " + ", ".join(f"{m}={row[m]:.4f}" for m in PYMDMA_METRICS))
    return rows


def analyse_embeddings(
    folder: str | Path,
    methods: list[str] = list(METHODS),
    balance: bool = True,
    max_per_group: int | None = None,
    seed: int = 42,
    k: int = 5,
    metric: str = "euclidean",
    n_workers: int = 4,
    out_dir: str | Path | None = None,
    run: Any = None,
) -> list[dict]:
    """Run the projections and pymdma metrics for the embeddings in ``folder``."""
    from da_eval.embedding_evaluation import EmbeddingEvaluator

    folder = Path(folder)
    manifest = load_manifest(folder)
    groups = ordered_groups(manifest)
    out_dir = Path(out_dir) if out_dir is not None else folder / "embedding_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = EmbeddingEvaluator({g["name"]: str(folder / g["embeddings"]) for g in groups})
    if balance:
        evaluator.balance(max_samples_per_group=max_per_group, random_state=seed)

    model = manifest.get("model_name", "model")
    plot_projections(
        evaluator, methods, f"{model} embeddings, source {manifest.get('source_label', manifest['source'])}",
        out_dir, balance, seed, run,
    )
    rows = compute_group_metrics(evaluator, groups, folder, out_dir, k, metric, n_workers)

    if rows:
        csv_path = out_dir / "embedding_metrics.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[*] Metrics -> {csv_path}")
        if run is not None:
            log_metrics(run, rows)
    return rows


def log_metrics(run: Any, rows: list[dict]) -> None:
    """``<dataset>/pymdma/<metric>`` summary keys (like ``<dataset>/map50`` in eval runs) + a table."""
    import wandb

    for row in rows:
        for m in PYMDMA_METRICS:
            if row[m] is not None:
                run.summary[f"{row['dataset']}/pymdma/{m}"] = row[m]
    columns = list(rows[0])
    run.log({"Table 3: Embedding distribution metrics": wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])})


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="da-eval embeddings",
        description="Projections and pymdma metrics for the embeddings of one source model "
                    "(reference: source train).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", help="Source results folder, e.g. results/rtdetr/source_AFO")
    src.add_argument("--artifact", help="W&B embeddings artifact, e.g. embeddings-rtdetr-C:latest (needs --wandb-project)")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), help="Projections to plot")
    p.add_argument("--no-balance", action="store_true", help="Plot all images instead of a balanced subset")
    p.add_argument("--max-per-group", type=int, default=None, help="Cap of images per group in the balanced plots")
    p.add_argument("--seed", type=int, default=42, help="Seed for balancing and projections")
    p.add_argument("--k", type=int, default=5, help="Nearest neighbours for the pymdma metrics")
    p.add_argument("--metric", default="euclidean", help="Distance for the pymdma metrics")
    p.add_argument("--n-workers", type=int, default=4, help="Workers for the pymdma metrics")
    p.add_argument("--output-dir", default=None, help="Default: <results>/embedding_analysis")
    p.add_argument("--wandb-project", default=None, help="Log to W&B (separate run in the eval run's group)")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-id", default=None, help="Add the results to this existing run (e.g. the eval run) instead")
    return p


def _artifact_source_group(artifact_path: str) -> str | None:
    """Group of the evaluation run that logged the artifact (so the analysis run joins it)."""
    try:
        import wandb
        try:
            from dotenv import load_dotenv
            load_dotenv(Path("~/.env").expanduser())
        except ImportError:
            pass
        wandb.login()
        producer = wandb.Api().artifact(artifact_path).logged_by()
        return producer.group if producer is not None else None
    except Exception as exc:
        print(f"[!] Could not look up the run that logged {artifact_path}: {exc}")
        return None


def main(argv: list[str] | None = None) -> None:
    from da_eval.pipeline import init_wandb_run

    args = build_parser().parse_args(argv)
    if args.artifact and not args.wandb_project:
        raise SystemExit("--artifact needs --wandb-project")
    settings = {
        "methods": args.methods, "balanced": not args.no_balance, "max_per_group": args.max_per_group,
        "seed": args.seed, "k": args.k, "metric": args.metric,
    }

    run = None
    if args.results:
        folder = Path(args.results)
        manifest = load_manifest(folder)
        group = f"{manifest.get('source_label', manifest['source'])}_metrics"   # same group as the eval run
        run = init_wandb_run(group, args.wandb_project, args.wandb_entity, ["embedding_analysis"],
                             job_type="embedding-analysis", name=f"embeddings_{group}", run_id=args.wandb_run_id,
                             config={**settings, "results": str(folder.resolve())})
        if args.wandb_project and run is None:
            raise SystemExit("W&B run could not be started (see message above).")
    else:
        path = args.artifact if args.artifact.count("/") == 2 else (
            f"{args.wandb_entity}/{args.wandb_project}/{args.artifact}" if args.wandb_entity
            else f"{args.wandb_project}/{args.artifact}")
        group = _artifact_source_group(path) or "embedding_analysis"
        run = init_wandb_run(group, args.wandb_project, args.wandb_entity, ["embedding_analysis"],
                             job_type="embedding-analysis", name=f"embeddings_{group}", run_id=args.wandb_run_id,
                             config={**settings, "artifact": args.artifact})
        if run is None:
            raise SystemExit("--artifact needs a working W&B run (see message above).")
        artifact = run.use_artifact(args.artifact)   # records lineage: this run used that artifact
        folder = Path(artifact.download())
        print(f"[*] Downloaded {args.artifact} -> {folder}")

    try:
        out_dir = args.output_dir or (folder / "embedding_analysis" if args.results else Path("embedding_analysis") / folder.name)
        analyse_embeddings(
            folder, methods=args.methods, balance=not args.no_balance, max_per_group=args.max_per_group,
            seed=args.seed, k=args.k, metric=args.metric, n_workers=args.n_workers, out_dir=out_dir, run=run,
        )
    finally:
        if run is not None:
            run.finish()

"""
Standalone experiment runner for DSPy evaluations.
Does not require the full Marin installation — only dspy, bm25s, orjson.

Usage on Colab:
    !pip install dspy bm25s orjson
    !python experiments/dspy/run_experiments.py \
        --task hover \
        --adapters chat baml toon \
        --split test \
        --max_examples 50 \
        --output_path /content/outputs \
        --api_key sk-...
"""

import argparse
import enum
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal ModelConfig (no Marin dependency)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    path: str | None = None
    engine_kwargs: dict = None

    def __post_init__(self):
        if self.engine_kwargs is None:
            self.engine_kwargs = {}


# ---------------------------------------------------------------------------
# Copy of _EnumSafeEncoder from dspy_evaluator
# ---------------------------------------------------------------------------

class _EnumSafeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


# ---------------------------------------------------------------------------
# Imports from experiments (no Marin base class needed)
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib" / "marin" / "src"))

from experiments.dspy.adapters.baml import BAMLAdapter
from experiments.dspy.adapters.toon import ToonAdapter
from experiments.dspy.programs.hover import HoVer, ClaimVerificationLabel
from experiments.dspy.programs.hotpotqa import HotpotQA


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _hover_metric(example, pred, trace=None):
    if pred is None:
        return 0.0
    gold_int = 1 if str(getattr(example, "label", "")).upper() == "SUPPORTED" else 0
    return float(gold_int == getattr(pred, "label_int", -1))


def _hotpotqa_metric(example, pred, trace=None):
    if pred is None:
        return 0.0
    gold = str(getattr(example, "answer", "")).lower().strip()
    predicted = str(getattr(pred, "answer", "") or "").lower().strip()
    return float(gold == predicted)


ADAPTER_MAP = {
    "baml": BAMLAdapter,
    "chat": dspy.ChatAdapter,
    "toon": ToonAdapter,
}

TASK_MAP = {
    "hover":    {"program": HoVer,     "metric": _hover_metric},
    "hotpotqa": {"program": HotpotQA,  "metric": _hotpotqa_metric},
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_hover(split, max_examples):
    from dspy.datasets.dataloader import DataLoader
    dl = DataLoader()
    full = dl.from_huggingface(
        dataset_name="vincentkoc/hover-parquet",
        split="train",
        trust_remote_code=True,
        fields=("claim", "label", "num_hops", "hpqa_id"),
        input_keys=("claim",),
    )
    hpqa_ids = set()
    full = [
        ex for ex in full
        if ex["num_hops"] == 2
        and ex["hpqa_id"] not in hpqa_ids
        and not hpqa_ids.add(ex["hpqa_id"])
    ]
    n = len(full)
    lo, hi = {"train": (0, int(0.8*n)), "dev": (int(0.8*n), int(0.9*n)), "test": (int(0.9*n), n)}[split]
    examples = full[lo:hi]
    if max_examples:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} HoVer examples (split={split})")
    return examples


def _load_hotpotqa(split, max_examples):
    from dspy.datasets.dataloader import DataLoader
    dl = DataLoader()
    full = dl.from_huggingface(
        dataset_name="hotpotqa/hotpot_qa",
        name="fullwiki",
        split="train",
        input_keys=("question",),
    )
    n = len(full)
    lo, hi = {"train": (0, int(0.8*n)), "dev": (int(0.8*n), int(0.9*n)), "test": (int(0.9*n), n)}[split]
    examples = full[lo:hi]
    if max_examples:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} HotpotQA examples (split={split})")
    return examples


# ---------------------------------------------------------------------------
# BM25S retriever
# ---------------------------------------------------------------------------

_WIKI_URL  = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
_WIKI_FILE = "wiki.abstracts.2017.jsonl"


def _build_bm25s_retriever():
    import bm25s
    import orjson
    from dspy.utils import download

    if not Path(_WIKI_FILE).exists():
        logger.info("Downloading Wikipedia corpus (~500 MB)...")
        download(_WIKI_URL)
        import subprocess
        subprocess.run(["tar", "-xzvf", "wiki.abstracts.2017.tar.gz"], check=True)

    logger.info("Loading corpus...")
    corpus = []
    with open(_WIKI_FILE) as f:
        for line in f:
            doc = orjson.loads(line)
            corpus.append(f"{doc['title']} | {' '.join(doc['text'])}")

    logger.info(f"Indexing {len(corpus):,} passages...")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    logger.info("BM25S ready.")

    def rm(query, k=3, **kwargs):
        tokens = bm25s.tokenize(query, show_progress=False)
        results, scores = retriever.retrieve(tokens, k=min(k, len(corpus)), n_threads=1, show_progress=False)
        return {corpus[idx]: float(sc) for idx, sc in zip(results[0], scores[0])}

    return rm


# ---------------------------------------------------------------------------
# Run one adapter
# ---------------------------------------------------------------------------

def run(model_name, api_key, endpoint, adapter_name, task_name, split, max_examples, output_path):
    task_cfg = TASK_MAP[task_name]

    lm = dspy.LM(
        model=f"openai/{model_name}",
        base_url=endpoint,
        api_key=api_key,
        temperature=0.0,
        cache=False,
    )
    adapter = ADAPTER_MAP[adapter_name]()
    dspy.configure(lm=lm, adapter=adapter)

    examples = _load_hover(split, max_examples) if task_name == "hover" else _load_hotpotqa(split, max_examples)
    rm = _build_bm25s_retriever()
    program = task_cfg["program"](search=rm)
    metric  = task_cfg["metric"]

    trajectories = []
    total_score = 0.0
    total_errors = 0
    start = time.time()

    for i, example in enumerate(examples):
        traj = {
            "sample_id": i,
            "input":     example.toDict() if hasattr(example, "toDict") else vars(example),
            "output":    None,
            "score":     None,
            "parsing_error": False,
            "evidence_with_scores": [],
        }
        try:
            pred  = program(**example.inputs())
            score = float(metric(example, pred))
            traj["output"] = pred.toDict() if hasattr(pred, "toDict") else str(pred)
            traj["score"]  = score
            passages = getattr(pred, "passages", None) or []
            if passages:
                traj["evidence_with_scores"] = passages
            total_score += score
        except Exception as exc:
            traj["parsing_error"] = True
            traj["error"] = str(exc)
            total_errors += 1
            logger.warning(f"Example {i} failed: {exc}")

        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            logger.info(f"{i+1}/{len(examples)} — running accuracy: {total_score/(i+1):.2%}")

    n = len(examples)
    results = {
        "task":              task_name,
        "adapter":           adapter_name,
        "split":             split,
        "model":             model_name,
        "total_examples":    n,
        "accuracy":          round(total_score / n, 4) if n > 0 else 0.0,
        "format_error_rate": round(total_errors / n, 4) if n > 0 else 0.0,
        "elapsed_seconds":   round(time.time() - start, 2),
    }

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task_name}_{adapter_name}_{split}"

    with open(out_dir / f"{stem}_trajectories.jsonl", "w") as f:
        for t in trajectories:
            f.write(json.dumps(t, cls=_EnumSafeEncoder, ensure_ascii=False) + "\n")

    with open(out_dir / f"{stem}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"{'='*50}")
    logger.info(f"  Task     : {results['task']}")
    logger.info(f"  Adapter  : {results['adapter']}")
    logger.info(f"  Accuracy : {results['accuracy']:.2%}")
    logger.info(f"  Errors   : {results['format_error_rate']:.2%}")
    logger.info(f"  Elapsed  : {results['elapsed_seconds']}s")
    logger.info(f"{'='*50}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="gpt-4o")
    parser.add_argument("--api_key",      required=True)
    parser.add_argument("--endpoint",     default=None)
    parser.add_argument("--task",         default="hover", choices=["hover", "hotpotqa"])
    parser.add_argument("--adapters",     nargs="+", default=["chat", "baml", "toon"])
    parser.add_argument("--split",        default="test")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--output_path",  default="outputs")
    args = parser.parse_args()

    all_results = {}
    for adapter in args.adapters:
        logger.info(f"\n>>> Running: {args.task} / {adapter}")
        r = run(
            model_name   = args.model,
            api_key      = args.api_key,
            endpoint     = args.endpoint,
            adapter_name = adapter,
            task_name    = args.task,
            split        = args.split,
            max_examples = args.max_examples,
            output_path  = args.output_path,
        )
        all_results[adapter] = r

    print("\n\nSUMMARY")
    print("="*50)
    for adapter, r in all_results.items():
        print(f"{adapter:10} accuracy={r['accuracy']:.2%}  errors={r['format_error_rate']:.2%}")

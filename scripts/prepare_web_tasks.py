"""Build the Web-service tasks as DatasetDicts under <datasets root>/<task>, the layout of the GLUE tasks.

banking77   Customer-support intent detection: PolyAI Banking77 (13,083 online-banking queries, 77 intents,
            CC BY 4.0).  train -> train, test -> validation (the official evaluation set).
moderation  Cross-platform content moderation.  Platform 0: Civil Comments (comments on news sites, CC0),
            toxic iff toxicity >= 0.5; platform 1: TweetEval-offensive (tweets, OffensEval 2019).  The
            "group" column (platform) drives the client partition: half of the clients per platform.
            Civil Comments is subsampled with a fixed seed; TweetEval train + validation are used for
            training.  Held-out sets: a Civil Comments test sample and the TweetEval test set, one split
            each, so the primary metric can weight the platforms equally.

Reads the raw files fetched by scripts/server/download_web_tasks.sh from --raw; downloads nothing.

Usage (server):  python scripts/prepare_web_tasks.py --raw <dir> --out <datasets root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

SEED = 20261015
CIVIL_TRAIN = 40_000
CIVIL_TEST = 10_000
TOXIC_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def banking77(raw: Path) -> tuple[DatasetDict, dict]:
    train = pd.read_csv(raw / "banking77" / "train.csv")
    test = pd.read_csv(raw / "banking77" / "test.csv")
    names = sorted(train["category"].unique())
    if sorted(test["category"].unique()) != names or len(names) != 77:
        raise ValueError("unexpected Banking77 categories")
    features = Features({"text": Value("string"), "label": ClassLabel(names=names)})

    def split(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {"text": frame["text"].astype(str).tolist(), "label": [names.index(c) for c in frame["category"]]},
            features=features,
        )

    data = DatasetDict({"train": split(train), "validation": split(test)})
    return data, {"rows": {k: len(v) for k, v in data.items()}, "classes": len(names)}


def moderation(raw: Path) -> tuple[DatasetDict, dict]:
    rng = np.random.default_rng(SEED)
    civil_train = pd.concat(
        [pd.read_parquet(path) for path in sorted((raw / "civil_comments").glob("train-*.parquet"))], ignore_index=True
    )
    civil_test = pd.read_parquet(raw / "civil_comments" / "test-00000-of-00001.parquet")
    tweet = {s: pd.read_parquet(raw / "tweet_eval_offensive" / f"{s}-00000-of-00001.parquet") for s in ("train", "validation", "test")}

    def civil(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        frame = frame[frame["text"].astype(str).str.strip().str.len() > 0]
        frame = frame.iloc[np.sort(rng.choice(len(frame), size=n, replace=False))]
        return pd.DataFrame({"text": frame["text"].astype(str), "label": (frame["toxicity"] >= TOXIC_THRESHOLD).astype(int), "group": 0})

    def twitter(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"text": frame["text"].astype(str), "label": frame["label"].astype(int), "group": 1})

    features = Features({"text": Value("string"), "label": ClassLabel(names=["acceptable", "toxic"]), "group": Value("int64")})

    def dataset(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(frame.reset_index(drop=True), features=features, preserve_index=False)

    data = DatasetDict(
        {
            "train": dataset(pd.concat([civil(civil_train, CIVIL_TRAIN), twitter(tweet["train"]), twitter(tweet["validation"])])),
            "validation_civil": dataset(civil(civil_test, CIVIL_TEST)),
            "validation_twitter": dataset(twitter(tweet["test"])),
        }
    )
    stats = {}
    for name, split in data.items():
        labels = np.asarray(split["label"])
        groups = np.asarray(split["group"])
        stats[name] = {
            "rows": len(split),
            "rows_by_platform": {int(g): int((groups == g).sum()) for g in np.unique(groups)},
            "toxic_rate_by_platform": {int(g): float(labels[groups == g].mean()) for g in np.unique(groups)},
        }
    return data, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    manifest = {
        "seed": SEED,
        "civil_train": CIVIL_TRAIN,
        "civil_test": CIVIL_TEST,
        "toxic_threshold": TOXIC_THRESHOLD,
        "raw_files": {str(p.relative_to(raw)): sha256(p) for p in sorted(raw.rglob("*")) if p.is_file()},
    }
    for task, build in (("banking77", banking77), ("moderation", moderation)):
        data, stats = build(raw)
        if (out / task).exists():
            raise SystemExit(f"{out / task} exists; remove it to rebuild")
        data.save_to_disk(str(out / task))
        manifest[task] = stats
        print(task, json.dumps(stats))
    (out / "web_tasks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

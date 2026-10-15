from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


SUPPORTED_METHODS = {
    "fedavg_lora",
    "ffa_lora",
    "flasc",
    "full_lora_ef",
    "lora_xs",
    "random_a_ef",
    "random_a_dense",
    "spectral_a_ef",
    "random_a_topk",
    "spectral_a_topk",
    "spectral_a_dense",
    "spectral_rot_ef",
    "spectral_block_ef",
    "hybrid4_a_ef",
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
    "hybrid4_a_decay_ef",
    "random_block_ef",
    "full_lora_dcef",
    "full_lora_dcef_client",
    "random_a_dcef",
    "spectral_a_dcef",
    "cab_a_dcef",
    "cab0_a_dcef",
    "cab_a_ef",
    "cab0_a_ef",
    "gev_a_ef",
    "gev0_a_ef",
    "cab0wh_a_ef",
    "randomwh_a_ef",
    "gev0x_a_ef",
    "gev0_a_dense",
    "eva_a_ef",
    "evawh_a_ef",
    # Published federated LoRA baselines (fedpeft.baselines).
    "flora_dense",
    "fedex_dense",
    "rolora_dense",
    "rolora_ef",
    "fedsvd_dense",
    "fedsvd_ef",
    "fedsb",
    "fedsb_norm",
}
BASELINE_DENSE_ONLY_METHODS = {"flora_dense", "fedex_dense", "rolora_dense", "fedsvd_dense"}
BASELINE_SPARSE_ONLY_METHODS = {"rolora_ef", "fedsvd_ef", "fedsb", "fedsb_norm"}
ACTIVATION_BASIS_METHODS = {
    "gev_a_ef",
    "gev0_a_ef",
    "cab0wh_a_ef",
    "randomwh_a_ef",
    "gev0x_a_ef",
    "gev0_a_dense",
    "eva_a_ef",
    "evawh_a_ef",
}
CAB_METHODS = {"cab_a_dcef", "cab0_a_dcef", "cab_a_ef", "cab0_a_ef", *ACTIVATION_BASIS_METHODS}
# provisional: smoke only; dcef_dev: median DCEF secant estimate; assumed: a
# declared prior (the DCEF secant estimate proved contaminated by client drift).
CAB_GAMMA_SOURCES = {"provisional", "dcef_dev", "assumed"}
SUPPORTED_TASKS = {"sst2", "rte", "mnli", "mrpc", "qqp", "qnli"}

STABILITY_PROTOCOL_SHA256 = (
    "e96302aca47dc5e5afba431ce1e02fb3d653fa6196d1ad09acf3594e980a5116"
)
STABILITY_V3_PROTOCOL_SHA256 = (
    "0a98550eeb4f915c60a132f1a199f49a7b6f8260ee5ab247e00b0964c4cb3b41"
)
STABILITY_V4_PROTOCOL_SHA256 = (
    "d22a2fd4d45717bb507f5572b9a6eabbc78332b6217cefcd36b1464fdc9df04c"
)
DIAGNOSTIC_RECIPE_VERSION = "v1"
STABILITY_CANDIDATES = (
    "spectral_rot_ef",
    "spectral_block_ef",
    "hybrid4_a_ef",
)
STABILITY_V3_CANDIDATE = "hybrid4_a_decay_ef"
STABILITY_V4_CANDIDATE = STABILITY_V3_CANDIDATE
STABILITY_V3_HYBRID_METHODS = {
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
    "hybrid4_a_ef",
    STABILITY_V3_CANDIDATE,
}
STABILITY_SPARSE_METHODS = {
    *STABILITY_CANDIDATES,
    STABILITY_V3_CANDIDATE,
    "random_block_ef",
}
MATCHED_FIXED_A_METHODS = {
    "random_a_dense",
    "spectral_a_dense",
    "random_a_topk",
    "spectral_a_topk",
    "random_a_ef",
    "spectral_a_ef",
    "random_a_dcef",
    "spectral_a_dcef",
    "cab_a_dcef",
    "cab0_a_dcef",
    "cab_a_ef",
    "cab0_a_ef",
    *ACTIVATION_BASIS_METHODS,
    *STABILITY_V3_HYBRID_METHODS,
    *STABILITY_SPARSE_METHODS,
}
MATCHED_FULL_LORA_METHODS = {
    "fedavg_lora",
    "full_lora_ef",
    "full_lora_dcef",
    "full_lora_dcef_client",
}


def matrix_run_count(cfg: dict[str, Any]) -> int:
    return len(cfg["data"]["tasks"]) * len(cfg["data"]["partition_seeds"]) * (
        len(cfg["matrix"]["dense_methods"])
        + len(cfg["matrix"]["sparse_methods"])
        * len(cfg["communication"]["wire_ratios"])
    )


def required_run_count(cfg: dict[str, Any]) -> int:
    """Return the configured final-matrix size, preserving the 81-run default."""
    return int(cfg["project"].get("expected_runs", 81))


def result_path(cfg: dict[str, Any], path: str | Path) -> Path:
    """Resolve a generated result path and keep it under docs/results."""
    project_root = Path(cfg["project"]["root"]).expanduser().resolve()
    results_root = (project_root / "docs" / "results").resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if candidate != results_root and results_root not in candidate.parents:
        raise ValueError(f"result path must be inside {results_root}: {candidate}")
    return candidate


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("project", "model", "data", "training", "communication", "matrix"):
        if section not in cfg:
            raise ValueError(f"missing config section: {section}")
    cfg["project"]["output_dir"] = str(result_path(cfg, cfg["project"]["output_dir"]))
    phase = cfg["project"].get("phase")
    if phase not in {"smoke", "tune", "final"}:
        raise ValueError("project.phase must be one of: smoke, tune, final")
    data = cfg["data"]
    unknown_tasks = sorted(set(data.get("tasks", [])) - SUPPORTED_TASKS)
    if unknown_tasks:
        raise ValueError(f"unsupported tasks: {unknown_tasks}")
    if not data.get("tasks"):
        raise ValueError("data.tasks must not be empty")
    if data["clients_per_round"] != data["num_clients"]:
        raise ValueError(
            "This implementation uses synchronized downlink error feedback and "
            "therefore requires full participation: clients_per_round == num_clients."
        )
    if not 0 < float(data["dirichlet_alpha"]):
        raise ValueError("dirichlet_alpha must be positive")
    if not 0 <= float(data["local_holdout_fraction"]) < 0.5:
        raise ValueError("local_holdout_fraction must be in [0, 0.5)")
    ratios = cfg["communication"]["wire_ratios"]
    if not ratios or any(not 0 < float(x) < 1 for x in ratios):
        raise ValueError("wire_ratios must contain values strictly between 0 and 1")
    evaluation_policy = cfg.get("evaluation", {}).get("policy", "official_validation")
    if evaluation_policy not in {"official_validation", "train_holdout"}:
        raise ValueError(
            "evaluation.policy must be either official_validation or train_holdout"
        )
    if phase == "tune" and evaluation_policy != "train_holdout":
        raise ValueError("tune phase must use train_holdout evaluation")
    if evaluation_policy == "train_holdout" and float(data["local_holdout_fraction"]) <= 0:
        raise ValueError("train_holdout evaluation requires a positive local_holdout_fraction")
    if phase == "final":
        seeds = [int(seed) for seed in data["partition_seeds"]]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("final phase requires exactly three distinct partition seeds")
        if evaluation_policy != "official_validation":
            raise ValueError("final phase must use official_validation evaluation")
        expected_runs = cfg["project"].get("expected_runs", 81)
        if (
            isinstance(expected_runs, bool)
            or not isinstance(expected_runs, int)
            or expected_runs <= 0
        ):
            raise ValueError("project.expected_runs must be a positive integer")
    methods: list[str] = []
    for key in (
        "dense_methods",
        "sparse_methods",
        "diagnostic_dense_methods",
        "diagnostic_sparse_methods",
    ):
        methods.extend(cfg["matrix"].get(key, []))
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    dense_methods = list(cfg["matrix"].get("dense_methods", []))
    sparse_methods = list(cfg["matrix"].get("sparse_methods", []))
    diagnostic_dense_methods = list(
        cfg["matrix"].get("diagnostic_dense_methods", [])
    )
    diagnostic_sparse_methods = list(
        cfg["matrix"].get("diagnostic_sparse_methods", [])
    )
    method_buckets = {
        "dense_methods": dense_methods,
        "sparse_methods": sparse_methods,
        "diagnostic_dense_methods": diagnostic_dense_methods,
        "diagnostic_sparse_methods": diagnostic_sparse_methods,
    }
    for key, values in method_buckets.items():
        if len(values) != len(set(values)):
            raise ValueError(f"matrix.{key} contains duplicate methods")
    memberships: dict[str, list[str]] = {}
    for key, values in method_buckets.items():
        for method in values:
            memberships.setdefault(method, []).append(key)
    duplicated = {
        method: buckets for method, buckets in memberships.items() if len(buckets) > 1
    }
    if duplicated:
        raise ValueError(f"methods occur in multiple matrix buckets: {duplicated}")
    wrongly_dense = sorted(
        STABILITY_SPARSE_METHODS.intersection(
            {*dense_methods, *diagnostic_dense_methods}
        )
    )
    if wrongly_dense:
        raise ValueError(
            f"stability EF methods must be configured as sparse: {wrongly_dense}"
        )
    if "random_block_ef" in sparse_methods:
        raise ValueError(
            "random_block_ef is diagnostic-only and must be listed only in "
            "matrix.diagnostic_sparse_methods"
        )
    stability = cfg.get("stability")
    if (
        stability is not None
        or STABILITY_SPARSE_METHODS.intersection(methods)
        or STABILITY_V3_HYBRID_METHODS.intersection(methods)
    ):
        _validate_stability_config(cfg, methods)
    if CAB_METHODS.intersection(methods) or "cab" in cfg:
        _validate_cab_config(cfg)
    misplaced = sorted(
        BASELINE_DENSE_ONLY_METHODS.intersection({*sparse_methods, *diagnostic_sparse_methods})
        | BASELINE_SPARSE_ONLY_METHODS.intersection({*dense_methods, *diagnostic_dense_methods})
    )
    if misplaced:
        raise ValueError(f"baseline methods in the wrong matrix bucket: {misplaced}")
    if "baselines" in cfg:
        unknown_fields = sorted(set(cfg["baselines"]) - {"fedsb_gradient_examples", "fedsb_gradient_batch_size"})
        if unknown_fields:
            raise ValueError(f"unsupported baselines fields: {unknown_fields}")
        for field in ("fedsb_gradient_examples", "fedsb_gradient_batch_size"):
            if field in cfg["baselines"] and int(cfg["baselines"][field]) < 1:
                raise ValueError(f"baselines.{field} must be positive")
    overrides = cfg["training"].get("method_overrides", {})
    unknown_override_methods = sorted(set(overrides) - SUPPORTED_METHODS)
    if unknown_override_methods:
        raise ValueError(f"unsupported method overrides: {unknown_override_methods}")
    for method, values in overrides.items():
        unknown_keys = sorted(set(values) - {"learning_rate", "head_learning_rate"})
        if unknown_keys:
            raise ValueError(f"unsupported training overrides for {method}: {unknown_keys}")
    schedules = cfg["training"].get("schedule_overrides", {})
    if not isinstance(schedules, dict):
        raise ValueError("training.schedule_overrides must be a mapping")
    unknown_schedule_methods = sorted(set(schedules) - SUPPORTED_METHODS)
    if unknown_schedule_methods:
        raise ValueError(
            f"unsupported schedule overrides: {unknown_schedule_methods}"
        )
    allowed_schedule_keys = {
        "name",
        "start_fraction",
        "final_multiplier",
        "apply_to",
    }
    for method, values in schedules.items():
        if not isinstance(values, dict):
            raise ValueError(f"schedule override for {method} must be a mapping")
        unknown_keys = sorted(set(values) - allowed_schedule_keys)
        if unknown_keys:
            raise ValueError(
                f"unsupported schedule override fields for {method}: {unknown_keys}"
            )
        missing_keys = sorted(allowed_schedule_keys - set(values))
        if missing_keys:
            raise ValueError(
                f"schedule override for {method} is missing fields: {missing_keys}"
            )
        if values.get("name") != "late_cosine_v1":
            raise ValueError(f"unsupported schedule override for {method}")
        start_fraction = float(values["start_fraction"])
        final_multiplier = float(values["final_multiplier"])
        if not math.isfinite(start_fraction) or not 0 < start_fraction < 1:
            raise ValueError(f"invalid schedule start_fraction for {method}")
        if not math.isfinite(final_multiplier) or not 0 < final_multiplier <= 1:
            raise ValueError(f"invalid schedule final_multiplier for {method}")
        if list(values.get("apply_to", [])) != ["adapter", "head"]:
            raise ValueError(
                f"schedule for {method} must apply to adapter and head"
            )
    for task in data["tasks"]:
        for method in set(methods):
            adapter_lr = float(method_task_value(cfg, "learning_rate", task, method))
            if not math.isfinite(adapter_lr) or adapter_lr <= 0:
                raise ValueError(f"learning rate must be positive for {task}/{method}")
            try:
                head_lr = float(method_task_value(cfg, "head_learning_rate", task, method))
            except KeyError:
                head_lr = adapter_lr
            if not math.isfinite(head_lr) or head_lr <= 0:
                raise ValueError(f"head learning rate must be positive for {task}/{method}")
        matched_fixed_a = MATCHED_FIXED_A_METHODS.intersection(methods)
        if len(matched_fixed_a) > 1:
            adapter_lrs = {
                float(method_task_value(cfg, "learning_rate", task, method))
                for method in matched_fixed_a
            }
            if len(adapter_lrs) != 1:
                raise ValueError(
                    f"matched fixed-A methods must share learning rate for {task}"
                )
            head_lrs = set()
            for method in matched_fixed_a:
                try:
                    head_lr = float(
                        method_task_value(cfg, "head_learning_rate", task, method)
                    )
                except KeyError:
                    head_lr = float(
                        method_task_value(cfg, "learning_rate", task, method)
                    )
                head_lrs.add(head_lr)
            if len(head_lrs) != 1:
                raise ValueError(
                    f"matched fixed-A methods must share head learning rate for {task}"
                )
        matched_full_lora = MATCHED_FULL_LORA_METHODS.intersection(methods)
        if len(matched_full_lora) > 1:
            adapter_lrs = {
                float(method_task_value(cfg, "learning_rate", task, method))
                for method in matched_full_lora
            }
            if len(adapter_lrs) != 1:
                raise ValueError(
                    f"matched full-LoRA methods must share learning rate for {task}"
                )
            head_lrs = set()
            for method in matched_full_lora:
                try:
                    head_lr = float(
                        method_task_value(cfg, "head_learning_rate", task, method)
                    )
                except KeyError:
                    head_lr = float(
                        method_task_value(cfg, "learning_rate", task, method)
                    )
                head_lrs.add(head_lr)
            if len(head_lrs) != 1:
                raise ValueError(
                    f"matched full-LoRA methods must share head learning rate for {task}"
                )
    communication = cfg["communication"]
    if communication.get("value_dtype") != "float16":
        raise ValueError("the implemented wire codec requires value_dtype: float16")
    if communication.get("index_dtype") != "uint32":
        raise ValueError("the implemented sparse codec requires index_dtype: uint32")
    if communication.get("count_downlink_per_client") is not True:
        raise ValueError("count_downlink_per_client must be true for delivered-wire reporting")


def _validate_cab_config(cfg: dict[str, Any]) -> None:
    """Validate the communication-aware basis settings (fedpeft.cab)."""
    section = cfg.get("cab")
    if not isinstance(section, dict):
        raise ValueError("CAB methods require a top-level cab section")
    required = {
        "gamma",
        "gamma_source",
        "client_rank",
        "gradient_examples",
        "gradient_batch_size",
    }
    allowed = required | {
        "kappa_multiplier",
        "kappa_override",
        "activation_rank",
        "activation_ridge",
        "factor_dtype",
        "basis_dtype",
        "basis_refine",
        "share_basis",
    }
    missing = sorted(required - set(section))
    if missing:
        raise ValueError(f"cab section is missing fields: {missing}")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unsupported cab fields: {unknown}")
    if section["gamma_source"] not in CAB_GAMMA_SOURCES:
        raise ValueError(f"cab.gamma_source must be one of {sorted(CAB_GAMMA_SOURCES)}")
    if section["gamma_source"] == "provisional" and cfg["project"].get("phase") != "smoke":
        raise ValueError(
            "cab.gamma is provisional; calibrate it from the DCEF development stage "
            "(scripts/calibrate_cab_gamma.py) before running outside smoke"
        )
    for task in cfg["data"]["tasks"]:
        gamma = float(task_value(cfg, "cab", "gamma", task))
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError(f"cab.gamma must lie in (0, 1] for {task}")
        for key in ("gradient_examples", "gradient_batch_size"):
            value = task_value(cfg, "cab", key, task)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"cab.{key} must be a positive integer for {task}")
    client_rank = section["client_rank"]
    rank = cfg.get("model", {}).get("rank")
    if isinstance(client_rank, bool) or int(client_rank) != client_rank or int(client_rank) < 1:
        raise ValueError("cab.client_rank must be a positive integer")
    # The server only needs the aggregate of the clients' low-rank reports to span
    # at least `rank` directions, so per-client rank can sit below the LoRA rank.
    num_clients = int(cfg.get("data", {}).get("num_clients", 1))
    if rank is not None and int(client_rank) * num_clients < int(rank):
        raise ValueError("cab.client_rank times data.num_clients must be at least the LoRA rank")
    multiplier = float(section.get("kappa_multiplier", 1.0))
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("cab.kappa_multiplier must be finite and non-negative")
    override = section.get("kappa_override")
    if override is not None and (not math.isfinite(float(override)) or float(override) < 0):
        raise ValueError("cab.kappa_override must be null or a non-negative number")
    activation_rank = section.get("activation_rank", 16)
    if isinstance(activation_rank, bool) or int(activation_rank) != activation_rank or int(activation_rank) < 1:
        raise ValueError("cab.activation_rank must be a positive integer")
    ridge = float(section.get("activation_ridge", 1e-3))
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("cab.activation_ridge must be a positive number")
    for key in ("factor_dtype", "basis_dtype"):
        if str(section.get(key, "float16")) not in {"float16", "int8"}:
            raise ValueError(f"cab.{key} must be float16 or int8")
    if str(section.get("basis_refine", "none")) not in {"none", "gram"}:
        raise ValueError("cab.basis_refine must be none or gram")
    if not isinstance(section.get("share_basis", False), bool):
        raise ValueError("cab.share_basis must be true or false")


def _validate_stability_config(cfg: dict[str, Any], methods: list[str]) -> None:
    """Validate the frozen post-hoc stability protocol without affecting legacy runs."""
    stability = cfg.get("stability")
    if not isinstance(stability, dict):
        raise ValueError("stability methods require a top-level stability section")
    if (
        stability.get("generation") == 4
        or stability.get("protocol_sha256") == STABILITY_V4_PROTOCOL_SHA256
    ):
        _validate_stability_v4_config(cfg, methods)
        return
    if (
        stability.get("generation") == 3
        or stability.get("protocol_sha256") == STABILITY_V3_PROTOCOL_SHA256
    ):
        _validate_stability_v3_config(cfg, methods)
        return
    required = {
        "stage",
        "protocol_sha256",
        "diagnostic_recipe_version",
        "candidates",
        "winner",
    }
    missing = sorted(required - set(stability))
    if missing:
        raise ValueError(f"stability section is missing fields: {missing}")
    if stability.get("protocol_sha256") != STABILITY_PROTOCOL_SHA256:
        raise ValueError(
            "stability.protocol_sha256 differs from the preregistered protocol"
        )
    if stability.get("diagnostic_recipe_version") != DIAGNOSTIC_RECIPE_VERSION:
        raise ValueError(
            "stability.diagnostic_recipe_version must be "
            f"{DIAGNOSTIC_RECIPE_VERSION}"
        )
    if tuple(stability.get("candidates", ())) != STABILITY_CANDIDATES:
        raise ValueError(
            "stability.candidates must preserve the frozen P1/P2/P3 order"
        )
    stage = stability.get("stage")
    if stage not in {"smoke", "stage_a", "stage_b", "confirmation"}:
        raise ValueError(
            "stability.stage must be one of: smoke, stage_a, stage_b, confirmation"
        )
    winner = stability.get("winner")
    if stage in {"smoke", "stage_a"}:
        if winner is not None:
            raise ValueError(f"stability.winner must be null during {stage}")
    elif winner not in STABILITY_CANDIDATES:
        raise ValueError(
            "stability.winner must be one of the frozen candidates after Stage A"
        )

    data = cfg["data"]
    matrix = cfg["matrix"]
    training = cfg["training"]
    ratios = [float(value) for value in cfg["communication"]["wire_ratios"]]
    seeds = [int(value) for value in data["partition_seeds"]]
    evaluation_policy = cfg.get("evaluation", {}).get(
        "policy", "official_validation"
    )
    for key, expected in {
        "min_client_examples": 24,
        "gpu_resident": True,
    }.items():
        if data.get(key) != expected:
            raise ValueError(f"stability protocol freezes data.{key}")
    if stage != "smoke":
        expected_schedule = {
            "rounds": {"sst2": 30, "rte": 50, "mnli": 50},
            "local_steps": {"sst2": 8, "rte": 8, "mnli": 16},
            "batch_size": {"sst2": 128, "rte": 16, "mnli": 128},
            "eval_every": {"sst2": 1, "rte": 1, "mnli": 5},
        }
        for key, expected_by_task in expected_schedule.items():
            for task in data["tasks"]:
                if task_value(cfg, "training", key, task) != expected_by_task[task]:
                    raise ValueError(
                        f"stability protocol freezes training.{key} for {task}"
                    )
        frozen_scalars = {
            "eval_batch_size": 512,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": "bfloat16",
            "checkpoint_every": 1,
        }
        for key, expected in frozen_scalars.items():
            if training.get(key) != expected:
                raise ValueError(f"stability protocol freezes training.{key}")
        expected_fixed_lr = {"sst2": 3e-4, "rte": 3e-4, "mnli": 1e-3}
        expected_fedavg_lr = {"sst2": 1e-3, "rte": 3e-4, "mnli": 1e-3}
        for task in data["tasks"]:
            for method in set(methods):
                expected_lr = (
                    expected_fixed_lr[task]
                    if method in MATCHED_FIXED_A_METHODS
                    else expected_fedavg_lr[task]
                    if method == "fedavg_lora"
                    else None
                )
                if expected_lr is None:
                    continue
                if float(method_task_value(cfg, "learning_rate", task, method)) != expected_lr:
                    raise ValueError(
                        f"stability protocol freezes adapter learning rate for {task}/{method}"
                    )
                try:
                    actual_head_lr = float(
                        method_task_value(cfg, "head_learning_rate", task, method)
                    )
                except KeyError:
                    actual_head_lr = float(
                        method_task_value(cfg, "learning_rate", task, method)
                    )
                if actual_head_lr != expected_lr:
                    raise ValueError(
                        f"stability protocol freezes head learning rate for {task}/{method}"
                    )
        model = cfg["model"]
        frozen_model = {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "target_suffixes": ["attention.self.query", "attention.self.value"],
            "torch_dtype": "float32",
            "basis_seed": 20260910,
        }
        for key, expected in frozen_model.items():
            if model.get(key) != expected:
                raise ValueError(f"stability protocol freezes model.{key}")
        frozen_data = {
            "num_clients": 10,
            "clients_per_round": 10,
            "dirichlet_alpha": 0.3,
            "local_holdout_fraction": 0.1,
            "max_length": 128,
        }
        for key, expected in frozen_data.items():
            if data.get(key) != expected:
                raise ValueError(f"stability protocol freezes data.{key}")
        if cfg["communication"].get("schema_once_per_run") is not True:
            raise ValueError(
                "stability protocol requires one counted schema per run"
            )
    if stage == "stage_a":
        if cfg["project"].get("phase") != "tune":
            raise ValueError("stability Stage A must use project.phase: tune")
        if evaluation_policy != "train_holdout":
            raise ValueError("stability Stage A must use train_holdout evaluation")
        if list(data["tasks"]) != ["sst2", "rte"]:
            raise ValueError("stability Stage A tasks must be exactly [sst2, rte]")
        if seeds != [321, 322, 323]:
            raise ValueError("stability Stage A seeds must be exactly [321, 322, 323]")
        if len(ratios) != 1 or not math.isclose(
            ratios[0], 0.1, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("stability Stage A must use only the 10% wire budget")
        if list(matrix.get("dense_methods", [])) != [
            "fedavg_lora",
            "spectral_a_dense",
        ]:
            raise ValueError(
                "stability Stage A dense methods must be FedAvg then spectral dense"
            )
        if list(matrix.get("sparse_methods", [])) != [
            "spectral_a_ef",
            *STABILITY_CANDIDATES,
        ]:
            raise ValueError(
                "stability Stage A sparse methods must be original Spectral-A+EF "
                "followed by frozen P1/P2/P3"
            )
        if list(matrix.get("diagnostic_dense_methods", [])):
            raise ValueError("stability Stage A has no diagnostic dense methods")
        if list(matrix.get("diagnostic_sparse_methods", [])) != [
            "random_block_ef"
        ]:
            raise ValueError(
                "random_block_ef must be the sole diagnostic-only Stage A method"
            )
        actual_runs = matrix_run_count(cfg)
        if actual_runs != 36 or cfg["project"].get("expected_runs") != 36:
            raise ValueError(
                "stability Stage A must contain exactly 36 required runs"
            )
    elif stage == "stage_b":
        if cfg["project"].get("phase") != "tune":
            raise ValueError("stability Stage B must use project.phase: tune")
        if evaluation_policy != "train_holdout":
            raise ValueError("stability Stage B must use train_holdout evaluation")
        if seeds != [321, 322, 323]:
            raise ValueError("stability Stage B seeds must be exactly [321, 322, 323]")
        substage = stability.get("substage")
        if substage not in {"sst_rte_20", "mnli_10_20"}:
            raise ValueError(
                "stability Stage B requires substage: sst_rte_20 or mnli_10_20"
            )
        if list(matrix.get("diagnostic_dense_methods", [])) or list(
            matrix.get("diagnostic_sparse_methods", [])
        ):
            raise ValueError(
                "stability Stage B submatrices cannot contain diagnostic methods"
            )
        if substage == "sst_rte_20":
            valid = (
                list(data["tasks"]) == ["sst2", "rte"]
                and ratios == [0.2]
                and list(matrix.get("dense_methods", [])) == []
                and list(matrix.get("sparse_methods", [])) == [winner]
                and matrix_run_count(cfg) == 6
                and cfg["project"].get("expected_runs") == 6
            )
        else:
            valid = (
                list(data["tasks"]) == ["mnli"]
                and ratios == [0.1, 0.2]
                and list(matrix.get("dense_methods", [])) == ["fedavg_lora"]
                and list(matrix.get("sparse_methods", [])) == [winner]
                and matrix_run_count(cfg) == 9
                and cfg["project"].get("expected_runs") == 9
            )
        if not valid:
            raise ValueError(
                f"stability Stage B {substage} matrix differs from the frozen 6/9-run submatrix"
            )
    elif stage == "confirmation":
        if cfg["project"].get("phase") != "final":
            raise ValueError("stability confirmation must use project.phase: final")
        if evaluation_policy != "official_validation":
            raise ValueError(
                "stability confirmation must use official_validation evaluation"
            )
        if list(data["tasks"]) != ["sst2", "rte", "mnli"]:
            raise ValueError(
                "stability confirmation tasks must be exactly [sst2, rte, mnli]"
            )
        if seeds != [501, 502, 503]:
            raise ValueError(
                "stability confirmation seeds must be exactly [501, 502, 503]"
            )
        if ratios != [0.1, 0.2]:
            raise ValueError(
                "stability confirmation must use the 10% and 20% wire budgets"
            )
        if list(matrix.get("dense_methods", [])) != ["fedavg_lora"]:
            raise ValueError(
                "stability confirmation must use only FedAvg-LoRA as dense reference"
            )
        if list(matrix.get("sparse_methods", [])) != [
            "spectral_a_ef",
            winner,
        ]:
            raise ValueError(
                "stability confirmation sparse methods must be original Spectral-A+EF "
                "and the frozen winner"
            )
        diagnostic_dense = list(matrix.get("diagnostic_dense_methods", []))
        diagnostic_sparse = list(matrix.get("diagnostic_sparse_methods", []))
        if diagnostic_dense:
            raise ValueError(
                "stability confirmation has no diagnostic dense methods"
            )
        allowed_diagnostic_sparse = (
            [[], ["random_block_ef"]]
            if winner == "spectral_block_ef"
            else [[]]
        )
        if diagnostic_sparse not in allowed_diagnostic_sparse:
            raise ValueError(
                "stability confirmation permits only the conditional "
                "random_block_ef diagnostic control"
            )
        if matrix_run_count(cfg) != 45 or cfg["project"].get(
            "expected_runs"
        ) != 45:
            raise ValueError(
                "stability confirmation must contain exactly 45 required runs"
            )


def _validate_stability_v3_config(
    cfg: dict[str, Any], methods: list[str]
) -> None:
    """Validate the frozen post-hoc Hybrid4 late-decay protocol."""
    stability = cfg.get("stability")
    assert isinstance(stability, dict)
    required = {
        "generation",
        "stage",
        "protocol_sha256",
        "diagnostic_recipe_version",
        "candidate",
        "controls",
        "winner",
    }
    missing = sorted(required - set(stability))
    if missing:
        raise ValueError(f"stability-v3 section is missing fields: {missing}")
    if stability.get("generation") != 3:
        raise ValueError("stability-v3 generation must be exactly 3")
    if stability.get("protocol_sha256") != STABILITY_V3_PROTOCOL_SHA256:
        raise ValueError(
            "stability-v3 protocol_sha256 differs from the preregistered protocol"
        )
    if stability.get("diagnostic_recipe_version") != DIAGNOSTIC_RECIPE_VERSION:
        raise ValueError(
            "stability-v3 diagnostic_recipe_version must be "
            f"{DIAGNOSTIC_RECIPE_VERSION}"
        )
    if stability.get("candidate") != STABILITY_V3_CANDIDATE:
        raise ValueError(
            f"stability-v3 candidate must be {STABILITY_V3_CANDIDATE}"
        )
    expected_controls = [
        "fedavg_lora",
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
        "hybrid4_a_ef",
    ]
    if list(stability.get("controls", [])) != expected_controls:
        raise ValueError("stability-v3 controls differ from the frozen order")

    stage = stability.get("stage")
    if stage not in {"smoke", "stage_a", "stage_b", "confirmation"}:
        raise ValueError(
            "stability-v3 stage must be one of: smoke, stage_a, stage_b, confirmation"
        )
    winner = stability.get("winner")
    if stage in {"smoke", "stage_a"}:
        if winner is not None:
            raise ValueError(f"stability-v3 winner must be null during {stage}")
    elif winner != STABILITY_V3_CANDIDATE:
        raise ValueError(
            f"stability-v3 winner must be {STABILITY_V3_CANDIDATE} after Stage A"
        )

    training = cfg["training"]
    schedules = training.get("schedule_overrides", {})
    expected_schedule = {
        "name": "late_cosine_v1",
        "start_fraction": 0.8,
        "final_multiplier": 0.1,
        "apply_to": ["adapter", "head"],
    }
    expected_schedules = {
        "hybrid4_a_decay_dense": expected_schedule,
        "hybrid4_a_decay_ef": expected_schedule,
    }
    if schedules != expected_schedules:
        raise ValueError(
            "stability-v3 requires the frozen late_cosine_v1 schedule only on "
            "the two decay methods"
        )

    data = cfg["data"]
    matrix = cfg["matrix"]
    ratios = [float(value) for value in cfg["communication"]["wire_ratios"]]
    seeds = [int(value) for value in data["partition_seeds"]]
    evaluation_policy = cfg.get("evaluation", {}).get(
        "policy", "official_validation"
    )
    if list(matrix.get("diagnostic_dense_methods", [])) or list(
        matrix.get("diagnostic_sparse_methods", [])
    ):
        raise ValueError("stability-v3 matrices cannot contain diagnostic-only runs")

    if stage == "smoke":
        valid = (
            cfg["project"].get("phase") == "smoke"
            and evaluation_policy == "train_holdout"
            and list(data["tasks"]) == ["sst2"]
            and seeds == [311]
            and len(ratios) == 1
            and math.isclose(ratios[0], 0.1, rel_tol=0.0, abs_tol=1e-12)
            and list(matrix.get("dense_methods", []))
            == ["hybrid4_a_dense", "hybrid4_a_decay_dense"]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", "hybrid4_a_decay_ef"]
            and matrix_run_count(cfg) == 4
            and cfg["project"].get("expected_runs") == 4
            and task_value(cfg, "training", "rounds", "sst2") == 2
            and task_value(cfg, "training", "local_steps", "sst2") == 1
            and task_value(cfg, "training", "batch_size", "sst2") == 16
            and task_value(cfg, "training", "eval_every", "sst2") == 1
            and cfg["model"].get("rank") == 8
            and cfg["model"].get("basis_seed") == 20260910
            and data.get("num_clients") == 2
            and data.get("clients_per_round") == 2
            and data.get("min_client_examples") == 24
            and cfg["communication"].get("schema_once_per_run") is True
        )
        if not valid:
            raise ValueError("stability-v3 smoke must preserve the frozen 4-run matrix")
        for method in STABILITY_V3_HYBRID_METHODS:
            for key in ("learning_rate", "head_learning_rate"):
                if float(method_task_value(cfg, key, "sst2", method)) != 3e-4:
                    raise ValueError(
                        f"stability-v3 smoke freezes {key} for {method}"
                    )
        return

    expected_schedule_by_task = {
        "rounds": {"sst2": 30, "rte": 50, "mnli": 50},
        "local_steps": {"sst2": 8, "rte": 8, "mnli": 16},
        "batch_size": {"sst2": 128, "rte": 16, "mnli": 128},
        "eval_every": {"sst2": 1, "rte": 1, "mnli": 5},
    }
    for key, expected_by_task in expected_schedule_by_task.items():
        for task in data["tasks"]:
            if task not in expected_by_task:
                raise ValueError(
                    f"stability-v3 task {task} is outside the frozen {stage} suite"
                )
            if task_value(cfg, "training", key, task) != expected_by_task[task]:
                raise ValueError(
                    f"stability-v3 freezes training.{key} for {task}"
                )
    for key, expected in {
        "eval_batch_size": 512,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "mixed_precision": "bfloat16",
        "checkpoint_every": 1,
    }.items():
        if training.get(key) != expected:
            raise ValueError(f"stability-v3 freezes training.{key}")

    expected_fixed_lr = {"sst2": 3e-4, "rte": 3e-4, "mnli": 1e-3}
    expected_fedavg_lr = {"sst2": 1e-3, "rte": 3e-4, "mnli": 1e-3}
    for task in data["tasks"]:
        for method in set(methods):
            if method in STABILITY_V3_HYBRID_METHODS:
                expected_lr = expected_fixed_lr[task]
            elif method == "fedavg_lora":
                expected_lr = expected_fedavg_lr[task]
            else:
                raise ValueError(f"method is outside stability-v3: {method}")
            for key in ("learning_rate", "head_learning_rate"):
                actual = float(method_task_value(cfg, key, task, method))
                if actual != expected_lr:
                    raise ValueError(
                        f"stability-v3 freezes {key} for {task}/{method}"
                    )

    for key, expected in {
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_suffixes": ["attention.self.query", "attention.self.value"],
        "torch_dtype": "float32",
        "basis_seed": 20260910,
    }.items():
        if cfg["model"].get(key) != expected:
            raise ValueError(f"stability-v3 freezes model.{key}")
    for key, expected in {
        "num_clients": 10,
        "clients_per_round": 10,
        "dirichlet_alpha": 0.3,
        "local_holdout_fraction": 0.1,
        "min_client_examples": 24,
        "max_length": 128,
        "gpu_resident": True,
    }.items():
        if data.get(key) != expected:
            raise ValueError(f"stability-v3 freezes data.{key}")
    if cfg["communication"].get("schema_once_per_run") is not True:
        raise ValueError("stability-v3 requires one counted schema per run")

    if stage == "stage_a":
        valid = (
            cfg["project"].get("phase") == "tune"
            and evaluation_policy == "train_holdout"
            and list(data["tasks"]) == ["sst2", "rte"]
            and seeds == [331, 332, 333]
            and len(ratios) == 1
            and math.isclose(ratios[0], 0.1, rel_tol=0.0, abs_tol=1e-12)
            and list(matrix.get("dense_methods", []))
            == [
                "fedavg_lora",
                "hybrid4_a_dense",
                "hybrid4_a_decay_dense",
            ]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", STABILITY_V3_CANDIDATE]
            and matrix_run_count(cfg) == 30
            and cfg["project"].get("expected_runs") == 30
        )
        if not valid:
            raise ValueError("stability-v3 Stage A must preserve the frozen 30-run matrix")
    elif stage == "stage_b":
        if cfg["project"].get("phase") != "tune" or evaluation_policy != "train_holdout":
            raise ValueError("stability-v3 Stage B must use tune/train_holdout")
        if seeds != [331, 332, 333]:
            raise ValueError("stability-v3 Stage B seeds must be 331/332/333")
        substage = stability.get("substage")
        if substage == "sst_rte_20":
            valid = (
                list(data["tasks"]) == ["sst2", "rte"]
                and ratios == [0.2]
                and list(matrix.get("dense_methods", [])) == []
                and list(matrix.get("sparse_methods", []))
                == [STABILITY_V3_CANDIDATE]
                and matrix_run_count(cfg) == 6
                and cfg["project"].get("expected_runs") == 6
            )
        elif substage == "mnli_10_20":
            valid = (
                list(data["tasks"]) == ["mnli"]
                and ratios == [0.1, 0.2]
                and list(matrix.get("dense_methods", [])) == ["fedavg_lora"]
                and list(matrix.get("sparse_methods", []))
                == [STABILITY_V3_CANDIDATE]
                and matrix_run_count(cfg) == 9
                and cfg["project"].get("expected_runs") == 9
            )
        else:
            valid = False
        if not valid:
            raise ValueError("stability-v3 Stage B matrix differs from the frozen submatrix")
    else:
        valid = (
            cfg["project"].get("phase") == "final"
            and evaluation_policy == "official_validation"
            and list(data["tasks"]) == ["sst2", "rte", "mnli"]
            and seeds == [501, 502, 503]
            and ratios == [0.1, 0.2]
            and list(matrix.get("dense_methods", [])) == ["fedavg_lora"]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", STABILITY_V3_CANDIDATE]
            and matrix_run_count(cfg) == 45
            and cfg["project"].get("expected_runs") == 45
        )
        if not valid:
            raise ValueError(
                "stability-v3 confirmation must preserve the frozen 45-run matrix"
            )


def _validate_stability_v4_config(
    cfg: dict[str, Any], methods: list[str]
) -> None:
    """Validate the preregistered MRPC/QQP task-suite revision."""
    stability = cfg.get("stability")
    assert isinstance(stability, dict)
    required = {
        "generation",
        "stage",
        "protocol_sha256",
        "diagnostic_recipe_version",
        "metric_schema",
        "candidate",
        "controls",
        "winner",
    }
    missing = sorted(required - set(stability))
    if missing:
        raise ValueError(f"stability-v4 section is missing fields: {missing}")
    if stability.get("generation") != 4:
        raise ValueError("stability-v4 generation must be exactly 4")
    if stability.get("protocol_sha256") != STABILITY_V4_PROTOCOL_SHA256:
        raise ValueError(
            "stability-v4 protocol_sha256 differs from the preregistered protocol"
        )
    if stability.get("diagnostic_recipe_version") != DIAGNOSTIC_RECIPE_VERSION:
        raise ValueError(
            "stability-v4 diagnostic_recipe_version must be "
            f"{DIAGNOSTIC_RECIPE_VERSION}"
        )
    if stability.get("metric_schema") != "glue-task-primary-v1":
        raise ValueError("stability-v4 metric_schema must be glue-task-primary-v1")
    if stability.get("candidate") != STABILITY_V4_CANDIDATE:
        raise ValueError(
            f"stability-v4 candidate must be {STABILITY_V4_CANDIDATE}"
        )
    expected_controls = [
        "fedavg_lora",
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
        "hybrid4_a_ef",
    ]
    if list(stability.get("controls", [])) != expected_controls:
        raise ValueError("stability-v4 controls differ from the frozen order")

    stage = stability.get("stage")
    if stage not in {"smoke", "stage_a", "stage_b", "confirmation"}:
        raise ValueError(
            "stability-v4 stage must be one of: smoke, stage_a, stage_b, confirmation"
        )
    winner = stability.get("winner")
    if stage in {"smoke", "stage_a"}:
        if winner is not None:
            raise ValueError(f"stability-v4 winner must be null during {stage}")
    elif winner != STABILITY_V4_CANDIDATE:
        raise ValueError(
            f"stability-v4 winner must be {STABILITY_V4_CANDIDATE} after Stage A"
        )

    expected_metrics = {
        "sst2": "accuracy",
        "mnli": "mean_matched_mismatched_accuracy",
        "mrpc": "glue_accuracy_f1_mean",
        "qqp": "glue_accuracy_f1_mean",
        "positive_label": 1,
    }
    if cfg.get("metrics") != expected_metrics:
        raise ValueError("stability-v4 metrics differ from the frozen task definitions")

    training = cfg["training"]
    schedules = training.get("schedule_overrides", {})
    expected_schedule = {
        "name": "late_cosine_v1",
        "start_fraction": 0.8,
        "final_multiplier": 0.1,
        "apply_to": ["adapter", "head"],
    }
    if schedules != {
        "hybrid4_a_decay_dense": expected_schedule,
        "hybrid4_a_decay_ef": expected_schedule,
    }:
        raise ValueError(
            "stability-v4 requires the frozen late_cosine_v1 schedule only on "
            "the two decay methods"
        )

    data = cfg["data"]
    matrix = cfg["matrix"]
    ratios = [float(value) for value in cfg["communication"]["wire_ratios"]]
    seeds = [int(value) for value in data["partition_seeds"]]
    evaluation_policy = cfg.get("evaluation", {}).get(
        "policy", "official_validation"
    )
    if list(matrix.get("diagnostic_dense_methods", [])) or list(
        matrix.get("diagnostic_sparse_methods", [])
    ):
        raise ValueError("stability-v4 matrices cannot contain diagnostic-only runs")

    if stage == "smoke":
        valid = (
            cfg["project"].get("phase") == "smoke"
            and evaluation_policy == "train_holdout"
            and list(data["tasks"]) == ["mrpc", "qqp"]
            and seeds == [311]
            and ratios == [0.1]
            and list(matrix.get("dense_methods", []))
            == ["fedavg_lora", "hybrid4_a_dense", "hybrid4_a_decay_dense"]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", STABILITY_V4_CANDIDATE]
            and matrix_run_count(cfg) == 10
            and cfg["project"].get("expected_runs") == 10
            and data.get("num_clients") == 2
            and data.get("clients_per_round") == 2
        )
        if not valid:
            raise ValueError("stability-v4 smoke must preserve the frozen 10-run matrix")
        expected_schedule_by_task = {
            "rounds": {"mrpc": 2, "qqp": 2},
            "local_steps": {"mrpc": 1, "qqp": 1},
            "batch_size": {"mrpc": 16, "qqp": 16},
            "eval_every": {"mrpc": 1, "qqp": 1},
        }
    else:
        expected_schedule_by_task = {
            "rounds": {"sst2": 30, "mrpc": 50, "qqp": 50, "mnli": 50},
            "local_steps": {"sst2": 8, "mrpc": 8, "qqp": 16, "mnli": 16},
            "batch_size": {"sst2": 128, "mrpc": 16, "qqp": 128, "mnli": 128},
            "eval_every": {"sst2": 1, "mrpc": 1, "qqp": 5, "mnli": 5},
        }
    for key, expected_by_task in expected_schedule_by_task.items():
        for task in data["tasks"]:
            if task not in expected_by_task:
                raise ValueError(
                    f"stability-v4 task {task} is outside the frozen {stage} suite"
                )
            if task_value(cfg, "training", key, task) != expected_by_task[task]:
                raise ValueError(f"stability-v4 freezes training.{key} for {task}")

    for key, expected in {
        "eval_batch_size": 512,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "mixed_precision": "bfloat16",
        "checkpoint_every": 1,
    }.items():
        if training.get(key) != expected:
            raise ValueError(f"stability-v4 freezes training.{key}")

    expected_lr = {"sst2": (1e-3, 3e-4), "mrpc": (3e-4, 3e-4), "qqp": (1e-3, 1e-3), "mnli": (1e-3, 1e-3)}
    for task in data["tasks"]:
        fedavg_lr, hybrid_lr = expected_lr[task]
        for method in set(methods):
            target_lr = fedavg_lr if method == "fedavg_lora" else hybrid_lr
            if method != "fedavg_lora" and method not in STABILITY_V3_HYBRID_METHODS:
                raise ValueError(f"method is outside stability-v4: {method}")
            for key in ("learning_rate", "head_learning_rate"):
                if float(method_task_value(cfg, key, task, method)) != target_lr:
                    raise ValueError(
                        f"stability-v4 freezes {key} for {task}/{method}"
                    )

    for key, expected in {
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_suffixes": ["attention.self.query", "attention.self.value"],
        "torch_dtype": "float32",
        "basis_seed": 20260910,
    }.items():
        if cfg["model"].get(key) != expected:
            raise ValueError(f"stability-v4 freezes model.{key}")
    for key, expected in {
        "dirichlet_alpha": 0.3,
        "local_holdout_fraction": 0.1,
        "min_client_examples": 24,
        "max_length": 128,
        "gpu_resident": True,
    }.items():
        if data.get(key) != expected:
            raise ValueError(f"stability-v4 freezes data.{key}")
    if stage != "smoke" and (
        data.get("num_clients") != 10 or data.get("clients_per_round") != 10
    ):
        raise ValueError("stability-v4 requires ten fully participating clients")
    if cfg["communication"].get("schema_once_per_run") is not True:
        raise ValueError("stability-v4 requires one counted schema per run")

    if stage == "smoke":
        return
    if stage == "stage_a":
        valid = (
            cfg["project"].get("phase") == "tune"
            and evaluation_policy == "train_holdout"
            and list(data["tasks"]) == ["sst2", "mrpc", "qqp"]
            and seeds == [351, 352, 353]
            and ratios == [0.1]
            and list(matrix.get("dense_methods", []))
            == ["fedavg_lora", "hybrid4_a_dense", "hybrid4_a_decay_dense"]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", STABILITY_V4_CANDIDATE]
            and matrix_run_count(cfg) == 45
            and cfg["project"].get("expected_runs") == 45
        )
        if not valid:
            raise ValueError("stability-v4 Stage A must preserve the frozen 45-run matrix")
    elif stage == "stage_b":
        if cfg["project"].get("phase") != "tune" or evaluation_policy != "train_holdout":
            raise ValueError("stability-v4 Stage B must use tune/train_holdout")
        if seeds != [351, 352, 353]:
            raise ValueError("stability-v4 Stage B seeds must be 351/352/353")
        substage = stability.get("substage")
        if substage == "primary_20":
            valid = (
                list(data["tasks"]) == ["sst2", "mrpc", "qqp"]
                and ratios == [0.2]
                and list(matrix.get("dense_methods", [])) == []
                and list(matrix.get("sparse_methods", []))
                == [STABILITY_V4_CANDIDATE]
                and matrix_run_count(cfg) == 9
                and cfg["project"].get("expected_runs") == 9
            )
        elif substage == "mnli_10_20":
            valid = (
                list(data["tasks"]) == ["mnli"]
                and ratios == [0.1, 0.2]
                and list(matrix.get("dense_methods", [])) == ["fedavg_lora"]
                and list(matrix.get("sparse_methods", []))
                == [STABILITY_V4_CANDIDATE]
                and matrix_run_count(cfg) == 9
                and cfg["project"].get("expected_runs") == 9
            )
        else:
            valid = False
        if not valid:
            raise ValueError("stability-v4 Stage B differs from the frozen submatrix")
    else:
        valid = (
            cfg["project"].get("phase") == "final"
            and evaluation_policy == "official_validation"
            and list(data["tasks"]) == ["sst2", "mnli", "mrpc", "qqp"]
            and seeds == [601, 602, 603]
            and ratios == [0.1, 0.2]
            and list(matrix.get("dense_methods", [])) == ["fedavg_lora"]
            and list(matrix.get("sparse_methods", []))
            == ["hybrid4_a_ef", STABILITY_V4_CANDIDATE]
            and matrix_run_count(cfg) == 60
            and cfg["project"].get("expected_runs") == 60
        )
        if not valid:
            raise ValueError(
                "stability-v4 confirmation must preserve the frozen 60-run matrix"
            )


def task_value(cfg: dict[str, Any], section: str, key: str, task: str) -> Any:
    value = cfg[section][key]
    return value[task] if isinstance(value, dict) else value


def method_task_value(cfg: dict[str, Any], key: str, task: str, method: str) -> Any:
    method_values = cfg["training"].get("method_overrides", {}).get(method, {})
    if key in method_values:
        value = method_values[key]
    else:
        value = cfg["training"][key]
    return value[task] if isinstance(value, dict) else value


def run_id(task: str, method: str, seed: int, wire_ratio: float | None) -> str:
    suffix = "dense" if wire_ratio is None else f"wire-{int(round(wire_ratio * 100)):02d}"
    return f"{task}__{method}__seed-{seed}__{suffix}"

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover - optional artifact helper
    joblib = None

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - runtime dependency guard
    torch = None
    DataLoader = None
    Dataset = object
TorchModuleBase = torch.nn.Module if torch is not None else object

try:
    from train_ae_baseline import (
        CompactConvAutoencoder,
        compute_reconstruction_errors,
        parameter_count,
        train_model,
    )
    from train_shallow_baselines import build_feature_config, compute_anomaly_scores, extract_features
except ModuleNotFoundError:  # pragma: no cover - package-style fallback
    from scripts.train_ae_baseline import (
        CompactConvAutoencoder,
        compute_reconstruction_errors,
        parameter_count,
        train_model,
    )
    from scripts.train_shallow_baselines import build_feature_config, compute_anomaly_scores, extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "paderborn"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "paderborn_unified_baselines"
DEFAULT_MODELS = ("ocsvm", "isolation_forest", "compact_ae", "resdilated_ae", "conv_vae", "deep_svdd")
EXPERIMENTAL_MODELS: tuple[str, ...] = ()
SUPPORTED_MODELS = (*DEFAULT_MODELS, *EXPERIMENTAL_MODELS)
THRESHOLD_RULES = ("percentile_99_5", "mean_plus_3std", "median_plus_4mad")
SUMMARY_FIELDNAMES = [
    "model",
    "seed",
    "threshold_rule",
    "status",
    "score_source",
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall_fault",
    "far",
    "false_alarm_rate",
    "threshold",
    "num_test_healthy",
    "num_test_fault",
    "num_predicted_anomalies",
    "num_true_anomalies",
    "metrics_path",
    "run_config_path",
    "error",
]
SUMMARY_BY_MODEL_FIELDNAMES = [
    "model",
    "threshold_rule",
    "status",
    "n_success",
    "n_error",
    "seeds_success",
    "seeds_error",
    "auroc_mean",
    "auroc_std",
    "auprc_mean",
    "auprc_std",
    "f1_mean",
    "f1_std",
    "precision_mean",
    "precision_std",
    "recall_fault_mean",
    "recall_fault_std",
    "far_mean",
    "far_std",
    "threshold_mean",
    "threshold_std",
    "num_test_healthy",
    "num_test_fault",
    "error",
]


@dataclass(frozen=True)
class ArrayPaths:
    train_healthy: Path
    val_healthy: Path
    test_healthy: Path
    test_fault: Path
    fault_labels: Path


@dataclass(frozen=True)
class DatasetInfo:
    paths: ArrayPaths
    preprocessing_config: dict[str, Any]
    window_size: int
    train_shape: tuple[int, int]
    val_shape: tuple[int, int]
    test_healthy_shape: tuple[int, int]
    test_fault_shape: tuple[int, int]
    fault_labels_shape: tuple[int, ...]


@dataclass(frozen=True)
class ScoreBundle:
    val_healthy_scores: np.ndarray
    test_healthy_scores: np.ndarray
    test_fault_scores: np.ndarray
    score_source: str
    model_settings: dict[str, Any]
    extra_artifacts: dict[str, str]


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    dataset: DatasetInfo
    device: Any
    batch_size: int
    command: str
    started_at: str


class TorchMemmapWindowDataset(Dataset):
    def __init__(self, path: Path) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for neural baseline evaluation.")
        self.path = path
        self.windows = np.load(path, mmap_mode="r")
        if self.windows.ndim != 2:
            raise ValueError(f"Expected a 2D window array at {path.as_posix()}, got {self.windows.shape}")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Any:
        window = np.array(self.windows[index], dtype=np.float32, copy=True)
        return torch.from_numpy(window).unsqueeze(0)


class DeepSVDDEncoder1D(TorchModuleBase):
    def __init__(self, embedding_dim: int = 64, dropout: float = 0.0) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Deep SVDD.")
        super().__init__()
        nn = torch.nn
        if embedding_dim <= 0:
            raise ValueError("--svdd-embedding-dim must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("--svdd-dropout must be in [0, 1).")
        self.embedding_dim = int(embedding_dim)
        self.dropout_rate = float(dropout)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 16, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(4, 32, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        self.projection = nn.Linear(64, int(embedding_dim), bias=False)

    def forward(self, inputs: Any) -> Any:
        features = self.features(inputs).flatten(start_dim=1)
        return self.projection(self.dropout(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified healthy-only Paderborn baseline evaluator.",
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to run, or 'all'.")
    parser.add_argument(
        "--threshold-rule",
        choices=(*THRESHOLD_RULES, "all"),
        default="percentile_99_5",
        help="Healthy-validation threshold rule to apply.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--skip-train-if-artifacts-exist", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-cpu", type=int, default=128)
    parser.add_argument("--batch-size-cuda", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional epoch override for smoke tests.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--svdd-embedding-dim", type=int, default=64)
    parser.add_argument("--svdd-lr", type=float, default=1e-3)
    parser.add_argument("--svdd-weight-decay", type=float, default=1e-6)
    parser.add_argument("--svdd-epochs", type=int, default=8)
    parser.add_argument("--svdd-dropout", type=float, default=0.0)

    parser.add_argument("--psd-band-count", type=int, default=5)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--ocsvm-nu", type=float, default=0.05)
    parser.add_argument("--ocsvm-max-iter", type=int, default=2000)
    parser.add_argument("--iforest-n-estimators", type=int, default=300)
    parser.add_argument("--iforest-max-samples", type=int, default=256)
    parser.add_argument("--iforest-n-jobs", type=int, default=1)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2)
            handle.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{name: row.get(name, "") for name in fieldnames} for row in rows])
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_numpy_array(path: Path, values: np.ndarray, dtype: np.dtype | type | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        array = np.asarray(values if dtype is None else np.asarray(values, dtype=dtype))
        with temp_path.open("wb") as handle:
            np.save(handle, array)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def format_command(parts: list[str]) -> str:
    formatted: list[str] = []
    for part in parts:
        text = str(part)
        if any(character.isspace() for character in text):
            formatted.append(f'"{text}"')
        else:
            formatted.append(text)
    return " ".join(formatted)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_array_paths(processed_root: Path) -> ArrayPaths:
    return ArrayPaths(
        train_healthy=processed_root / "train" / "healthy_windows.npy",
        val_healthy=processed_root / "val" / "healthy_windows.npy",
        test_healthy=processed_root / "test" / "healthy_windows.npy",
        test_fault=processed_root / "test" / "fault_windows.npy",
        fault_labels=processed_root / "test" / "fault_labels.npy",
    )


def load_memmap(path: Path, expected_width: int | None = None) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D window array at {path.as_posix()}, got {array.shape}")
    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError(f"Expected width {expected_width} at {path.as_posix()}, got {array.shape}")
    return array


def discover_dataset(processed_root: Path, metadata_root: Path) -> DatasetInfo:
    paths = resolve_array_paths(processed_root)
    required = [
        paths.train_healthy,
        paths.val_healthy,
        paths.test_healthy,
        paths.test_fault,
        paths.fault_labels,
        metadata_root / "preprocessing_config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Paderborn inputs are missing: " + ", ".join(path.as_posix() for path in missing))

    preprocessing_config = read_json(metadata_root / "preprocessing_config.json")
    window_size = int(preprocessing_config.get("window_size", 2048))
    train = load_memmap(paths.train_healthy, window_size)
    val = load_memmap(paths.val_healthy, window_size)
    test_healthy = load_memmap(paths.test_healthy, window_size)
    test_fault = load_memmap(paths.test_fault, window_size)
    fault_labels = np.load(paths.fault_labels, mmap_mode="r")
    if fault_labels.ndim != 1:
        raise ValueError(f"Expected 1D fault_labels.npy, got {fault_labels.shape}")
    if fault_labels.shape[0] != test_fault.shape[0]:
        raise ValueError(
            f"fault_labels length differs from fault windows: {fault_labels.shape[0]} != {test_fault.shape[0]}"
        )
    if train.shape[0] == 0 or val.shape[0] == 0:
        raise RuntimeError("Healthy train and validation arrays must be non-empty.")
    if test_healthy.shape[0] == 0 or test_fault.shape[0] == 0:
        raise RuntimeError("Healthy and fault test arrays must be non-empty.")

    return DatasetInfo(
        paths=paths,
        preprocessing_config=preprocessing_config,
        window_size=window_size,
        train_shape=tuple(int(value) for value in train.shape),
        val_shape=tuple(int(value) for value in val.shape),
        test_healthy_shape=tuple(int(value) for value in test_healthy.shape),
        test_fault_shape=tuple(int(value) for value in test_fault.shape),
        fault_labels_shape=tuple(int(value) for value in fault_labels.shape),
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def select_device(device_name: str) -> Any:
    if torch is None:
        if device_name == "cuda":
            raise RuntimeError("PyTorch is not installed; CUDA device cannot be selected.")
        return "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective_batch_size(args: argparse.Namespace, device: Any) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    if torch is not None and isinstance(device, torch.device) and device.type == "cuda":
        return int(args.batch_size_cuda)
    return int(args.batch_size_cpu)


def effective_epoch_count(configured_epochs: int, max_epochs: int | None) -> int:
    if configured_epochs <= 0:
        raise ValueError("Epoch count must be positive.")
    if max_epochs is None:
        return int(configured_epochs)
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be positive when supplied.")
    return int(min(configured_epochs, max_epochs))


def expand_models(models: list[str]) -> list[str]:
    if "all" in models:
        return list(DEFAULT_MODELS)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}")
    return list(dict.fromkeys(models))


def expand_threshold_rules(rule: str) -> list[str]:
    if rule == "all":
        return list(THRESHOLD_RULES)
    return [rule]


def expand_seeds(args: argparse.Namespace) -> list[int]:
    seeds = args.seeds if args.seeds is not None else [args.seed]
    return [int(seed) for seed in dict.fromkeys(seeds)]


def threshold_run_dir(output_root: Path, model_name: str, seed: int, threshold_rule: str) -> Path:
    return output_root / model_name / f"seed_{seed}" / threshold_rule


def model_seed_dir(output_root: Path, model_name: str, seed: int) -> Path:
    return output_root / model_name / f"seed_{seed}"


def required_run_outputs(run_dir: Path) -> list[Path]:
    return [
        run_dir / "metrics.json",
        run_dir / "val_healthy_scores.npy",
        run_dir / "test_scores.npy",
        run_dir / "test_labels.npy",
        run_dir / "run_config.json",
    ]


def load_existing_summary_row(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "run_config.json")
    return {
        "model": metrics.get("model", config.get("model", "")),
        "seed": metrics.get("seed", config.get("seed", "")),
        "threshold_rule": metrics.get("threshold_rule", config.get("threshold_rule", "")),
        "status": "reused",
        "score_source": config.get("score_source", ""),
        "auroc": metrics.get("auroc", ""),
        "auprc": metrics.get("auprc", ""),
        "f1": metrics.get("f1", ""),
        "precision": metrics.get("precision", ""),
        "recall_fault": metrics.get("recall_fault", ""),
        "far": metrics.get("far", ""),
        "false_alarm_rate": metrics.get("false_alarm_rate", ""),
        "threshold": metrics.get("threshold", ""),
        "num_test_healthy": metrics.get("num_test_healthy", ""),
        "num_test_fault": metrics.get("num_test_fault", ""),
        "num_predicted_anomalies": metrics.get("num_predicted_anomalies", ""),
        "num_true_anomalies": metrics.get("num_true_anomalies", ""),
        "metrics_path": (run_dir / "metrics.json").as_posix(),
        "run_config_path": (run_dir / "run_config.json").as_posix(),
        "error": "",
    }


def all_threshold_outputs_complete(output_root: Path, model_name: str, seed: int, threshold_rules: list[str]) -> bool:
    for rule in threshold_rules:
        run_dir = threshold_run_dir(output_root, model_name, seed, rule)
        if not all(path.exists() for path in required_run_outputs(run_dir)):
            return False
    return True


def load_existing_rows(output_root: Path, model_name: str, seed: int, threshold_rules: list[str]) -> list[dict[str, Any]]:
    return [load_existing_summary_row(threshold_run_dir(output_root, model_name, seed, rule)) for rule in threshold_rules]


def extract_features_chunked(windows: np.ndarray, feature_config: Any, chunk_size: int, split_name: str) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("--feature-chunk-size must be positive.")
    row_count = int(windows.shape[0])
    feature_count = len(feature_config.feature_names)
    features = np.empty((row_count, feature_count), dtype=np.float32)
    log(f"Extracting {feature_count} features for {split_name} ({row_count} windows)")
    for start in range(0, row_count, chunk_size):
        end = min(start + chunk_size, row_count)
        features[start:end] = extract_features(np.asarray(windows[start:end], dtype=np.float32), feature_config)
    return features


def get_shallow_feature_bundle(context: RunContext, cache: dict[str, Any]) -> dict[str, Any]:
    cache_key = f"bands={context.args.psd_band_count};chunk={context.args.feature_chunk_size}"
    if cache_key in cache:
        return cache[cache_key]

    paths = context.dataset.paths
    window_size = context.dataset.window_size
    feature_config = build_feature_config(window_size, context.args.psd_band_count)
    train_windows = load_memmap(paths.train_healthy, window_size)
    val_windows = load_memmap(paths.val_healthy, window_size)
    test_healthy_windows = load_memmap(paths.test_healthy, window_size)
    test_fault_windows = load_memmap(paths.test_fault, window_size)
    bundle = {
        "feature_config": feature_config,
        "train": extract_features_chunked(train_windows, feature_config, context.args.feature_chunk_size, "train_healthy"),
        "val": extract_features_chunked(val_windows, feature_config, context.args.feature_chunk_size, "val_healthy"),
        "test_healthy": extract_features_chunked(
            test_healthy_windows,
            feature_config,
            context.args.feature_chunk_size,
            "test_healthy",
        ),
        "test_fault": extract_features_chunked(test_fault_windows, feature_config, context.args.feature_chunk_size, "test_fault"),
    }
    cache[cache_key] = bundle
    return bundle


def run_ocsvm(context: RunContext, seed: int, feature_cache: dict[str, Any]) -> ScoreBundle:
    bundle = get_shallow_feature_bundle(context, feature_cache)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(bundle["train"])
    val_features = scaler.transform(bundle["val"])
    test_healthy_features = scaler.transform(bundle["test_healthy"])
    test_fault_features = scaler.transform(bundle["test_fault"])

    model = SGDOneClassSVM(
        nu=context.args.ocsvm_nu,
        random_state=seed,
        max_iter=context.args.ocsvm_max_iter,
        tol=1e-3,
        shuffle=True,
    )
    log("Fitting OC-SVM on healthy training features")
    model.fit(train_features)
    seed_dir = model_seed_dir(context.args.output_root, "ocsvm", seed)
    artifact_paths: dict[str, str] = {}
    if joblib is not None:
        model_path = seed_dir / "ocsvm_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": tuple(bundle["feature_config"].feature_names),
                "settings": {"nu": context.args.ocsvm_nu, "max_iter": context.args.ocsvm_max_iter},
            },
            model_path,
        )
        artifact_paths["joblib"] = model_path.as_posix()

    return ScoreBundle(
        val_healthy_scores=compute_anomaly_scores(model, val_features),
        test_healthy_scores=compute_anomaly_scores(model, test_healthy_features),
        test_fault_scores=compute_anomaly_scores(model, test_fault_features),
        score_source="trained_sgd_one_class_svm",
        model_settings={
            "model_variant": "SGDOneClassSVM_linear_full_train",
            "nu": context.args.ocsvm_nu,
            "max_iter": context.args.ocsvm_max_iter,
            "feature_names": tuple(bundle["feature_config"].feature_names),
            "feature_scaler_fit": "train_healthy",
        },
        extra_artifacts=artifact_paths,
    )


def run_isolation_forest(context: RunContext, seed: int, feature_cache: dict[str, Any]) -> ScoreBundle:
    bundle = get_shallow_feature_bundle(context, feature_cache)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(bundle["train"])
    val_features = scaler.transform(bundle["val"])
    test_healthy_features = scaler.transform(bundle["test_healthy"])
    test_fault_features = scaler.transform(bundle["test_fault"])

    max_samples = min(int(context.args.iforest_max_samples), int(train_features.shape[0]))
    model = IsolationForest(
        n_estimators=context.args.iforest_n_estimators,
        max_samples=max_samples,
        contamination="auto",
        random_state=seed,
        n_jobs=context.args.iforest_n_jobs,
    )
    log("Fitting Isolation Forest on healthy training features")
    model.fit(train_features)
    seed_dir = model_seed_dir(context.args.output_root, "isolation_forest", seed)
    artifact_paths: dict[str, str] = {}
    if joblib is not None:
        model_path = seed_dir / "isolation_forest_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": tuple(bundle["feature_config"].feature_names),
                "settings": {
                    "n_estimators": context.args.iforest_n_estimators,
                    "max_samples": max_samples,
                    "n_jobs": context.args.iforest_n_jobs,
                },
            },
            model_path,
        )
        artifact_paths["joblib"] = model_path.as_posix()

    return ScoreBundle(
        val_healthy_scores=compute_anomaly_scores(model, val_features),
        test_healthy_scores=compute_anomaly_scores(model, test_healthy_features),
        test_fault_scores=compute_anomaly_scores(model, test_fault_features),
        score_source="trained_isolation_forest",
        model_settings={
            "model_variant": "IsolationForest",
            "n_estimators": context.args.iforest_n_estimators,
            "max_samples": max_samples,
            "n_jobs": context.args.iforest_n_jobs,
            "feature_names": tuple(bundle["feature_config"].feature_names),
            "feature_scaler_fit": "train_healthy",
        },
        extra_artifacts=artifact_paths,
    )


def require_torch_available() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for this neural baseline.")


def make_torch_loader(path: Path, batch_size: int, shuffle: bool, seed: int) -> Any:
    require_torch_available()
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        TorchMemmapWindowDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def load_compact_checkpoint_if_available(context: RunContext, seed: int) -> tuple[Any | None, str, dict[str, str]]:
    require_torch_available()
    candidates = [
        (model_seed_dir(context.args.output_root, "compact_ae", seed) / "compact_ae.pt", "unified_compact_ae_checkpoint"),
        (ARTIFACTS_ROOT / "models" / "paderborn_ae_baseline.pt", "existing_project_paderborn_ae_checkpoint"),
    ]
    for path, source in candidates:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(path, map_location="cpu")
        payload_seed = payload.get("seed")
        if source != "unified_compact_ae_checkpoint":
            if payload_seed is not None and int(payload_seed) != int(seed):
                continue
            if payload_seed is None and int(seed) != 42:
                continue
        state_dict = payload.get("model_state_dict") or payload.get("state_dict")
        if state_dict is None:
            continue
        model = CompactConvAutoencoder()
        model.load_state_dict(state_dict)
        model = model.to(context.device)
        model.eval()
        return model, source, {"checkpoint": path.as_posix()}
    return None, "", {}


def run_compact_ae(context: RunContext, seed: int, _feature_cache: dict[str, Any]) -> ScoreBundle:
    require_torch_available()
    checkpoint_source = ""
    extra_artifacts: dict[str, str] = {}
    model = None
    if context.args.skip_train_if_artifacts_exist:
        model, checkpoint_source, extra_artifacts = load_compact_checkpoint_if_available(context, seed)

    train_loader = make_torch_loader(context.dataset.paths.train_healthy, context.batch_size, shuffle=True, seed=seed)
    val_loader = make_torch_loader(context.dataset.paths.val_healthy, context.batch_size, shuffle=False, seed=seed)
    test_healthy_loader = make_torch_loader(context.dataset.paths.test_healthy, context.batch_size, shuffle=False, seed=seed)
    test_fault_loader = make_torch_loader(context.dataset.paths.test_fault, context.batch_size, shuffle=False, seed=seed)

    if model is None:
        log("Training CompactAE on healthy training windows")
        model = CompactConvAutoencoder().to(context.device)
        epochs = effective_epoch_count(context.args.epochs, context.args.max_epochs)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=context.device,
            epochs=epochs,
            learning_rate=context.args.learning_rate,
        )
        checkpoint_source = "trained_compact_ae"
        checkpoint_path = model_seed_dir(context.args.output_root, "compact_ae", seed) / "compact_ae.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": "CompactConvAutoencoder",
                "model_state_dict": model.state_dict(),
                "seed": int(seed),
                "epochs": int(epochs),
                "learning_rate": float(context.args.learning_rate),
                "history": history,
                "parameter_count": int(parameter_count(model)),
                "data_protocol": "healthy_train_only",
            },
            checkpoint_path,
        )
        extra_artifacts["checkpoint"] = checkpoint_path.as_posix()
        extra_artifacts["history_epochs"] = str(len(history.get("train_loss", [])))
    else:
        log(f"Loaded CompactAE from {checkpoint_source}")

    return ScoreBundle(
        val_healthy_scores=compute_reconstruction_errors(model, val_loader, context.device),
        test_healthy_scores=compute_reconstruction_errors(model, test_healthy_loader, context.device),
        test_fault_scores=compute_reconstruction_errors(model, test_fault_loader, context.device),
        score_source=checkpoint_source,
        model_settings={
            "model_variant": "CompactConvAutoencoder",
            "epochs": effective_epoch_count(context.args.epochs, context.args.max_epochs),
            "learning_rate": context.args.learning_rate,
            "batch_size": context.batch_size,
            "device": str(context.device),
            "parameter_count": int(parameter_count(model)),
        },
        extra_artifacts=extra_artifacts,
    )


def load_score_array(path: Path, expected_count: int, label: str) -> np.ndarray:
    values = np.load(path)
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if scores.shape[0] != expected_count:
        raise ValueError(f"{label} score count mismatch for {path.as_posix()}: {scores.shape[0]} != {expected_count}")
    return scores


def resdilated_source_paths(seed: int) -> dict[str, Path]:
    run_dir = ARTIFACTS_ROOT / "generative_upgrades" / "resdilated_ae" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "val_healthy_scores": run_dir / "val_healthy_scores.npy",
        "test_healthy_scores": run_dir / "test_healthy_scores.npy",
        "test_fault_scores": run_dir / "test_fault_scores.npy",
        "checkpoint": run_dir / "checkpoints" / "best.pt",
    }


def run_resdilated_from_saved_scores(context: RunContext, seed: int) -> ScoreBundle | None:
    paths = resdilated_source_paths(seed)
    score_paths = [paths["val_healthy_scores"], paths["test_healthy_scores"], paths["test_fault_scores"]]
    if not all(path.exists() for path in score_paths):
        return None
    log(f"Loading ResDilatedAE saved score arrays for seed {seed}")
    return ScoreBundle(
        val_healthy_scores=load_score_array(paths["val_healthy_scores"], context.dataset.val_shape[0], "val_healthy"),
        test_healthy_scores=load_score_array(
            paths["test_healthy_scores"],
            context.dataset.test_healthy_shape[0],
            "test_healthy",
        ),
        test_fault_scores=load_score_array(paths["test_fault_scores"], context.dataset.test_fault_shape[0], "test_fault"),
        score_source="existing_resdilated_ae_score_arrays",
        model_settings={
            "model_variant": "ResDilatedAE",
            "score_artifact_source": paths["run_dir"].as_posix(),
            "training_source": "artifacts/generative_upgrades/resdilated_ae",
        },
        extra_artifacts={key: path.as_posix() for key, path in paths.items() if path.exists()},
    )


def run_resdilated_from_checkpoint(context: RunContext, seed: int) -> ScoreBundle:
    require_torch_available()
    paths = resdilated_source_paths(seed)
    checkpoint = paths["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Missing ResDilatedAE saved score arrays and checkpoint for seed "
            f"{seed}: {checkpoint.as_posix()}"
        )

    try:
        from train_generative_upgrades import build_models, compute_reconstruction_scores, load_torch_payload, select_models
    except ModuleNotFoundError:  # pragma: no cover - package-style fallback
        from scripts.train_generative_upgrades import build_models, compute_reconstruction_scores, load_torch_payload, select_models

    payload = load_torch_payload(checkpoint)
    checkpoint_seed = payload.get("seed")
    if checkpoint_seed is not None and int(checkpoint_seed) != int(seed):
        raise RuntimeError(f"ResDilatedAE checkpoint seed mismatch: expected {seed}, found {checkpoint_seed}")
    settings = payload.get("training_settings", {})
    dropout = float(settings.get("dropout", 0.05))
    candidates = select_models(build_models(context.dataset.window_size, dropout), "resdilated_ae")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one ResDilatedAE model definition, found {len(candidates)}")
    _run_config, model = candidates[0]
    model.load_state_dict(payload["state_dict"])
    model = model.to(context.device)
    model.eval()

    val_loader = make_torch_loader(context.dataset.paths.val_healthy, context.batch_size, shuffle=False, seed=seed)
    test_healthy_loader = make_torch_loader(context.dataset.paths.test_healthy, context.batch_size, shuffle=False, seed=seed)
    test_fault_loader = make_torch_loader(context.dataset.paths.test_fault, context.batch_size, shuffle=False, seed=seed)
    log(f"Scoring ResDilatedAE from checkpoint for seed {seed}")
    return ScoreBundle(
        val_healthy_scores=compute_reconstruction_scores(model=model, loader=val_loader, device=context.device, model_kind="ae"),
        test_healthy_scores=compute_reconstruction_scores(
            model=model,
            loader=test_healthy_loader,
            device=context.device,
            model_kind="ae",
        ),
        test_fault_scores=compute_reconstruction_scores(
            model=model,
            loader=test_fault_loader,
            device=context.device,
            model_kind="ae",
        ),
        score_source="existing_resdilated_ae_checkpoint_inference",
        model_settings={
            "model_variant": "ResDilatedAE",
            "checkpoint_training_settings": settings,
            "batch_size": context.batch_size,
            "device": str(context.device),
        },
        extra_artifacts={"checkpoint": checkpoint.as_posix()},
    )


def run_resdilated_ae(context: RunContext, seed: int, _feature_cache: dict[str, Any]) -> ScoreBundle:
    saved = run_resdilated_from_saved_scores(context, seed)
    if saved is not None:
        return saved
    return run_resdilated_from_checkpoint(context, seed)


def conv_vae_source_paths(seed: int) -> dict[str, Path]:
    run_dir = ARTIFACTS_ROOT / "generative_upgrades" / "conv_vae" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "val_healthy_scores": run_dir / "val_healthy_scores.npy",
        "test_healthy_scores": run_dir / "test_healthy_scores.npy",
        "test_fault_scores": run_dir / "test_fault_scores.npy",
        "checkpoint": run_dir / "checkpoints" / "best.pt",
        "metrics_json": run_dir / "metrics.json",
        "status_json": run_dir / "status.json",
        "report_md": run_dir / "report.md",
    }


def run_conv_vae_from_saved_scores(context: RunContext, seed: int) -> ScoreBundle | None:
    paths = conv_vae_source_paths(seed)
    score_paths = [paths["val_healthy_scores"], paths["test_healthy_scores"], paths["test_fault_scores"]]
    if not all(path.exists() for path in score_paths):
        return None
    log(f"Loading ConvVAE saved score arrays for seed {seed}")
    return ScoreBundle(
        val_healthy_scores=load_score_array(paths["val_healthy_scores"], context.dataset.val_shape[0], "val_healthy"),
        test_healthy_scores=load_score_array(
            paths["test_healthy_scores"],
            context.dataset.test_healthy_shape[0],
            "test_healthy",
        ),
        test_fault_scores=load_score_array(paths["test_fault_scores"], context.dataset.test_fault_shape[0], "test_fault"),
        score_source="existing_conv_vae_score_arrays",
        model_settings={
            "model_variant": "ConvVAE",
            "score_artifact_source": paths["run_dir"].as_posix(),
            "training_source": "artifacts/generative_upgrades/conv_vae",
        },
        extra_artifacts={key: path.as_posix() for key, path in paths.items() if path.exists()},
    )


def run_conv_vae_from_checkpoint(context: RunContext, seed: int) -> ScoreBundle:
    paths = conv_vae_source_paths(seed)
    checkpoint = paths["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Missing ConvVAE saved score arrays and checkpoint for seed "
            f"{seed}; training required before unified evaluation. Expected checkpoint: {checkpoint.as_posix()}"
        )
    require_torch_available()

    try:
        from train_generative_upgrades import build_models, compute_reconstruction_scores, load_torch_payload, select_models
    except ModuleNotFoundError:  # pragma: no cover - package-style fallback
        from scripts.train_generative_upgrades import build_models, compute_reconstruction_scores, load_torch_payload, select_models

    payload = load_torch_payload(checkpoint)
    checkpoint_seed = payload.get("seed")
    if checkpoint_seed is not None and int(checkpoint_seed) != int(seed):
        raise RuntimeError(f"ConvVAE checkpoint seed mismatch: expected {seed}, found {checkpoint_seed}")
    checkpoint_model = payload.get("model_cli_name")
    if checkpoint_model is not None and checkpoint_model != "conv_vae":
        raise RuntimeError(f"ConvVAE checkpoint model mismatch: expected conv_vae, found {checkpoint_model}")

    settings = payload.get("training_settings", {})
    dropout = float(settings.get("dropout", 0.05))
    candidates = select_models(build_models(context.dataset.window_size, dropout), "conv_vae")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one ConvVAE model definition, found {len(candidates)}")
    _run_config, model = candidates[0]
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"ConvVAE checkpoint has no state dict: {checkpoint.as_posix()}")
    model.load_state_dict(state_dict)
    model = model.to(context.device)
    model.eval()

    val_loader = make_torch_loader(context.dataset.paths.val_healthy, context.batch_size, shuffle=False, seed=seed)
    test_healthy_loader = make_torch_loader(context.dataset.paths.test_healthy, context.batch_size, shuffle=False, seed=seed)
    test_fault_loader = make_torch_loader(context.dataset.paths.test_fault, context.batch_size, shuffle=False, seed=seed)
    log(f"Scoring ConvVAE from checkpoint for seed {seed}")
    return ScoreBundle(
        val_healthy_scores=compute_reconstruction_scores(model=model, loader=val_loader, device=context.device, model_kind="vae"),
        test_healthy_scores=compute_reconstruction_scores(
            model=model,
            loader=test_healthy_loader,
            device=context.device,
            model_kind="vae",
        ),
        test_fault_scores=compute_reconstruction_scores(
            model=model,
            loader=test_fault_loader,
            device=context.device,
            model_kind="vae",
        ),
        score_source="existing_conv_vae_checkpoint_inference",
        model_settings={
            "model_variant": "ConvVAE",
            "checkpoint_training_settings": settings,
            "batch_size": context.batch_size,
            "device": str(context.device),
        },
        extra_artifacts={key: path.as_posix() for key, path in paths.items() if path.exists()},
    )


def run_conv_vae(context: RunContext, seed: int, _feature_cache: dict[str, Any]) -> ScoreBundle:
    saved = run_conv_vae_from_saved_scores(context, seed)
    if saved is not None:
        return saved
    return run_conv_vae_from_checkpoint(context, seed)


def deep_svdd_paths(output_root: Path, seed: int) -> dict[str, Path]:
    seed_dir = model_seed_dir(output_root, "deep_svdd", seed)
    return {
        "seed_dir": seed_dir,
        "checkpoint": seed_dir / "deep_svdd.pt",
        "val_healthy_scores": seed_dir / "val_healthy_scores.npy",
        "test_healthy_scores": seed_dir / "test_healthy_scores.npy",
        "test_fault_scores": seed_dir / "test_fault_scores.npy",
    }


def deep_svdd_settings(context: RunContext) -> dict[str, Any]:
    return {
        "model_variant": "DeepSVDDEncoder1D",
        "embedding_dim": int(context.args.svdd_embedding_dim),
        "learning_rate": float(context.args.svdd_lr),
        "weight_decay": float(context.args.svdd_weight_decay),
        "epochs": int(effective_epoch_count(context.args.svdd_epochs, context.args.max_epochs)),
        "configured_epochs": int(context.args.svdd_epochs),
        "max_epochs": context.args.max_epochs,
        "dropout": float(context.args.svdd_dropout),
        "batch_size": int(context.batch_size),
        "device": str(context.device),
        "center_init": "mean_healthy_train_embeddings_before_svdd_optimization",
        "center_eps": 0.1,
        "loss": "mean_squared_distance_to_fixed_center",
        "best_checkpoint_selection": "lowest_healthy_validation_mean_score",
        "score_direction": "larger_is_more_abnormal",
    }


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    require_torch_available()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")


def load_deep_svdd_saved_scores(context: RunContext, seed: int) -> ScoreBundle | None:
    paths = deep_svdd_paths(context.args.output_root, seed)
    score_paths = [paths["val_healthy_scores"], paths["test_healthy_scores"], paths["test_fault_scores"]]
    if not all(path.exists() for path in score_paths):
        return None
    log(f"Loading Deep SVDD saved split score arrays for seed {seed}")
    return ScoreBundle(
        val_healthy_scores=load_score_array(paths["val_healthy_scores"], context.dataset.val_shape[0], "val_healthy"),
        test_healthy_scores=load_score_array(
            paths["test_healthy_scores"],
            context.dataset.test_healthy_shape[0],
            "test_healthy",
        ),
        test_fault_scores=load_score_array(paths["test_fault_scores"], context.dataset.test_fault_shape[0], "test_fault"),
        score_source="existing_deep_svdd_scores",
        model_settings=deep_svdd_settings(context),
        extra_artifacts={key: path.as_posix() for key, path in paths.items() if path.exists()},
    )


def compute_deep_svdd_center(model: Any, loader: Any, device: Any, eps: float = 0.1) -> Any:
    model.eval()
    embedding_sum = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            embeddings = model(batch).detach()
            batch_sum = embeddings.double().sum(dim=0)
            embedding_sum = batch_sum if embedding_sum is None else embedding_sum + batch_sum
            count += int(embeddings.shape[0])
    if embedding_sum is None or count == 0:
        raise RuntimeError("Cannot initialize Deep SVDD center from an empty healthy training loader.")
    center = (embedding_sum / float(count)).float()
    near_zero = center.abs() < float(eps)
    if bool(near_zero.any()):
        center[near_zero] = torch.where(center[near_zero] < 0, -torch.full_like(center[near_zero], eps), torch.full_like(center[near_zero], eps))
    return center.to(device)


def compute_deep_svdd_scores(model: Any, loader: Any, center: Any, device: Any) -> np.ndarray:
    model.eval()
    center = center.to(device).view(1, -1)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            embeddings = model(batch)
            distances = torch.sum((embeddings - center) ** 2, dim=1)
            scores.append(distances.detach().cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def deep_svdd_checkpoint_payload(
    *,
    model: Any,
    center: Any,
    seed: int,
    settings: dict[str, Any],
    history: dict[str, Any],
    best_epoch: int,
    best_validation_mean_score: float,
) -> dict[str, Any]:
    return {
        "model_name": "DeepSVDDEncoder1D",
        "model_state_dict": model.state_dict(),
        "center": center.detach().cpu(),
        "seed": int(seed),
        "embedding_dim": int(settings["embedding_dim"]),
        "model_settings": {
            "model_variant": "DeepSVDDEncoder1D",
            "embedding_dim": int(settings["embedding_dim"]),
            "dropout": float(settings["dropout"]),
            "parameter_count": int(parameter_count(model)),
        },
        "training_settings": {
            "learning_rate": float(settings["learning_rate"]),
            "weight_decay": float(settings["weight_decay"]),
            "epochs": int(settings["epochs"]),
            "batch_size": int(settings["batch_size"]),
            "device": str(settings["device"]),
            "center_init": settings["center_init"],
            "center_eps": float(settings["center_eps"]),
            "loss": settings["loss"],
            "best_checkpoint_selection": settings["best_checkpoint_selection"],
        },
        "history": history,
        "best_epoch": int(best_epoch),
        "best_validation_mean_score": float(best_validation_mean_score),
        "parameter_count": int(parameter_count(model)),
        "data_protocol": "healthy_train_only",
    }


def save_deep_svdd_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def train_deep_svdd_model(
    *,
    model: Any,
    center: Any,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    seed: int,
    settings: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history: dict[str, Any] = {
        "train_loss": [],
        "val_mean_score": [],
        "epoch_seconds": [],
    }
    best_epoch = 0
    best_validation_mean_score = math.inf
    best_state: dict[str, Any] | None = None
    center = center.to(device)
    for epoch in range(1, int(settings["epochs"]) + 1):
        epoch_started = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(batch)
            distances = torch.sum((embeddings - center.view(1, -1)) ** 2, dim=1)
            loss = distances.mean()
            loss.backward()
            optimizer.step()
            batch_count = int(batch.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        train_loss = total_loss / float(max(total_count, 1))
        val_scores = compute_deep_svdd_scores(model, val_loader, center, device)
        val_mean_score = float(np.mean(val_scores)) if val_scores.size else math.inf
        epoch_seconds = time.time() - epoch_started
        history["train_loss"].append(float(train_loss))
        history["val_mean_score"].append(float(val_mean_score))
        history["epoch_seconds"].append(float(epoch_seconds))
        log(
            f"Deep SVDD epoch {epoch}/{settings['epochs']}: "
            f"train_loss={train_loss:.6g} val_mean_score={val_mean_score:.6g}"
        )
        if val_mean_score < best_validation_mean_score:
            best_epoch = int(epoch)
            best_validation_mean_score = float(val_mean_score)
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            payload = deep_svdd_checkpoint_payload(
                model=model,
                center=center,
                seed=seed,
                settings=settings,
                history=history,
                best_epoch=best_epoch,
                best_validation_mean_score=best_validation_mean_score,
            )
            save_deep_svdd_checkpoint(checkpoint_path, payload)
    if best_state is None:
        raise RuntimeError("Deep SVDD training completed without a best checkpoint.")
    model.load_state_dict(best_state)
    payload = deep_svdd_checkpoint_payload(
        model=model,
        center=center,
        seed=seed,
        settings=settings,
        history=history,
        best_epoch=best_epoch,
        best_validation_mean_score=best_validation_mean_score,
    )
    save_deep_svdd_checkpoint(checkpoint_path, payload)
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_validation_mean_score": float(best_validation_mean_score),
    }


def load_deep_svdd_checkpoint(context: RunContext, seed: int, checkpoint_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    payload = load_torch_checkpoint(checkpoint_path)
    payload_seed = payload.get("seed")
    if payload_seed is not None and int(payload_seed) != int(seed):
        raise RuntimeError(f"Deep SVDD checkpoint seed mismatch: expected {seed}, found {payload_seed}")
    embedding_dim = int(payload.get("embedding_dim", payload.get("model_settings", {}).get("embedding_dim", 64)))
    dropout = float(payload.get("model_settings", {}).get("dropout", context.args.svdd_dropout))
    model = DeepSVDDEncoder1D(embedding_dim=embedding_dim, dropout=dropout).to(context.device)
    state_dict = payload.get("model_state_dict") or payload.get("encoder_state_dict") or payload.get("state_dict")
    if state_dict is None:
        raise RuntimeError(f"Deep SVDD checkpoint has no encoder state dict: {checkpoint_path.as_posix()}")
    center = payload.get("center")
    if center is None:
        raise RuntimeError(f"Deep SVDD checkpoint has no center: {checkpoint_path.as_posix()}")
    if not torch.is_tensor(center):
        center = torch.as_tensor(center, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()
    center = center.to(context.device, dtype=torch.float32)
    if int(center.reshape(-1).shape[0]) != embedding_dim:
        raise RuntimeError(f"Deep SVDD center dimension mismatch: {center.reshape(-1).shape[0]} != {embedding_dim}")
    return model, center.reshape(-1), payload


def save_deep_svdd_split_scores(paths: dict[str, Path], bundle: ScoreBundle) -> None:
    save_numpy_array(paths["val_healthy_scores"], bundle.val_healthy_scores, np.float32)
    save_numpy_array(paths["test_healthy_scores"], bundle.test_healthy_scores, np.float32)
    save_numpy_array(paths["test_fault_scores"], bundle.test_fault_scores, np.float32)


def run_deep_svdd(context: RunContext, seed: int, _feature_cache: dict[str, Any]) -> ScoreBundle:
    require_torch_available()
    settings = deep_svdd_settings(context)
    paths = deep_svdd_paths(context.args.output_root, seed)
    if context.args.skip_train_if_artifacts_exist:
        saved = load_deep_svdd_saved_scores(context, seed)
        if saved is not None:
            return saved
        if paths["checkpoint"].exists():
            log(f"Loading Deep SVDD checkpoint for seed {seed}")
            model, center, payload = load_deep_svdd_checkpoint(context, seed, paths["checkpoint"])
            val_loader = make_torch_loader(context.dataset.paths.val_healthy, context.batch_size, shuffle=False, seed=seed)
            test_healthy_loader = make_torch_loader(
                context.dataset.paths.test_healthy,
                context.batch_size,
                shuffle=False,
                seed=seed,
            )
            test_fault_loader = make_torch_loader(context.dataset.paths.test_fault, context.batch_size, shuffle=False, seed=seed)
            bundle = ScoreBundle(
                val_healthy_scores=compute_deep_svdd_scores(model, val_loader, center, context.device),
                test_healthy_scores=compute_deep_svdd_scores(model, test_healthy_loader, center, context.device),
                test_fault_scores=compute_deep_svdd_scores(model, test_fault_loader, center, context.device),
                score_source="loaded_deep_svdd_checkpoint",
                model_settings={
                    **settings,
                    "checkpoint_training_settings": payload.get("training_settings", {}),
                    "parameter_count": int(parameter_count(model)),
                },
                extra_artifacts={"checkpoint": paths["checkpoint"].as_posix()},
            )
            save_deep_svdd_split_scores(paths, bundle)
            return bundle

    log("Training Deep SVDD on healthy training windows")
    model = DeepSVDDEncoder1D(
        embedding_dim=int(context.args.svdd_embedding_dim),
        dropout=float(context.args.svdd_dropout),
    ).to(context.device)
    center_loader = make_torch_loader(context.dataset.paths.train_healthy, context.batch_size, shuffle=False, seed=seed)
    train_loader = make_torch_loader(context.dataset.paths.train_healthy, context.batch_size, shuffle=True, seed=seed)
    val_loader = make_torch_loader(context.dataset.paths.val_healthy, context.batch_size, shuffle=False, seed=seed)
    test_healthy_loader = make_torch_loader(context.dataset.paths.test_healthy, context.batch_size, shuffle=False, seed=seed)
    test_fault_loader = make_torch_loader(context.dataset.paths.test_fault, context.batch_size, shuffle=False, seed=seed)
    center = compute_deep_svdd_center(model, center_loader, context.device, eps=float(settings["center_eps"]))
    training_summary = train_deep_svdd_model(
        model=model,
        center=center,
        train_loader=train_loader,
        val_loader=val_loader,
        device=context.device,
        seed=seed,
        settings=settings,
        checkpoint_path=paths["checkpoint"],
    )
    if paths["checkpoint"].exists():
        model, center, _payload = load_deep_svdd_checkpoint(context, seed, paths["checkpoint"])
    bundle = ScoreBundle(
        val_healthy_scores=compute_deep_svdd_scores(model, val_loader, center, context.device),
        test_healthy_scores=compute_deep_svdd_scores(model, test_healthy_loader, center, context.device),
        test_fault_scores=compute_deep_svdd_scores(model, test_fault_loader, center, context.device),
        score_source="trained_deep_svdd",
        model_settings={
            **settings,
            "parameter_count": int(parameter_count(model)),
            "best_epoch": int(training_summary["best_epoch"]),
            "best_validation_mean_score": float(training_summary["best_validation_mean_score"]),
        },
        extra_artifacts={
            "checkpoint": paths["checkpoint"].as_posix(),
            "history_epochs": str(len(training_summary["history"].get("train_loss", []))),
        },
    )
    save_deep_svdd_split_scores(paths, bundle)
    bundle.extra_artifacts.update(
        {key: path.as_posix() for key, path in paths.items() if key.endswith("_scores") and path.exists()}
    )
    return bundle


def calibrate_threshold(val_healthy_scores: np.ndarray, rule: str) -> dict[str, Any]:
    scores = np.asarray(val_healthy_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise RuntimeError("Validation healthy scores are empty; cannot calibrate a threshold.")
    if rule == "percentile_99_5":
        threshold = float(np.percentile(scores, 99.5))
        return {
            "rule": rule,
            "threshold": threshold,
            "percentile": 99.5,
            "fit_split": "val_healthy",
            "fit_count": int(scores.size),
        }
    if rule == "mean_plus_3std":
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        return {
            "rule": rule,
            "threshold": float(score_mean + (3.0 * score_std)),
            "validation_mean": score_mean,
            "validation_std": score_std,
            "fit_split": "val_healthy",
            "fit_count": int(scores.size),
        }
    if rule == "median_plus_4mad":
        score_median = float(np.median(scores))
        score_mad = float(np.median(np.abs(scores - score_median)))
        return {
            "rule": rule,
            "threshold": float(score_median + (4.0 * score_mad)),
            "validation_median": score_median,
            "validation_mad": score_mad,
            "mad_definition": "raw_median_absolute_deviation",
            "fit_split": "val_healthy",
            "fit_count": int(scores.size),
        }
    raise ValueError(f"Unsupported threshold rule: {rule}")


def evaluate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != values.shape[0]:
        raise ValueError(f"Label/score length mismatch: {labels.shape[0]} != {values.shape[0]}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Binary test_labels must contain both healthy=0 and fault=1.")
    predictions = (values >= float(threshold)).astype(np.int64)
    healthy_mask = labels == 0
    fault_mask = labels == 1
    false_positives = int(((predictions == 1) & healthy_mask).sum())
    true_positives = int(((predictions == 1) & fault_mask).sum())
    num_healthy = int(healthy_mask.sum())
    num_fault = int(fault_mask.sum())
    far = float(false_positives / num_healthy) if num_healthy else 0.0
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_fault": float(recall_score(labels, predictions, zero_division=0)),
        "far": far,
        "false_alarm_rate": far,
        "threshold": float(threshold),
        "num_test_healthy": num_healthy,
        "num_test_fault": num_fault,
        "num_predicted_anomalies": int(predictions.sum()),
        "num_true_anomalies": int(labels.sum()),
        "false_positives_healthy": false_positives,
        "true_positives_fault": true_positives,
    }


def build_binary_test_labels(num_healthy: int, num_fault: int) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(int(num_healthy), dtype=np.int64),
            np.ones(int(num_fault), dtype=np.int64),
        ]
    )


def validate_score_bundle(bundle: ScoreBundle, dataset: DatasetInfo) -> None:
    expected = {
        "val_healthy": (bundle.val_healthy_scores, dataset.val_shape[0]),
        "test_healthy": (bundle.test_healthy_scores, dataset.test_healthy_shape[0]),
        "test_fault": (bundle.test_fault_scores, dataset.test_fault_shape[0]),
    }
    for label, (scores, count) in expected.items():
        if np.asarray(scores).reshape(-1).shape[0] != count:
            raise ValueError(f"{label} scores have wrong count: {np.asarray(scores).shape[0]} != {count}")


def save_threshold_artifacts(
    *,
    context: RunContext,
    model_name: str,
    seed: int,
    threshold_rule: str,
    bundle: ScoreBundle,
) -> dict[str, Any]:
    run_dir = threshold_run_dir(context.args.output_root, model_name, seed, threshold_rule)
    val_scores = np.asarray(bundle.val_healthy_scores, dtype=np.float32).reshape(-1)
    test_scores = np.concatenate(
        [
            np.asarray(bundle.test_healthy_scores, dtype=np.float32).reshape(-1),
            np.asarray(bundle.test_fault_scores, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32, copy=False)
    test_labels = build_binary_test_labels(bundle.test_healthy_scores.shape[0], bundle.test_fault_scores.shape[0])
    threshold_meta = calibrate_threshold(val_scores, threshold_rule)
    metrics = evaluate_metrics(test_labels, test_scores, float(threshold_meta["threshold"]))
    metrics_payload = {
        "model": model_name,
        "seed": int(seed),
        "threshold_rule": threshold_rule,
        **metrics,
        "threshold_meta": threshold_meta,
        "score_source": bundle.score_source,
    }
    run_config = {
        "study": "paderborn_unified_baselines",
        "model": model_name,
        "seed": int(seed),
        "threshold_rule": threshold_rule,
        "timestamp": context.started_at,
        "command": context.command,
        "score_source": bundle.score_source,
        "model_settings": bundle.model_settings,
        "extra_artifacts": bundle.extra_artifacts,
        "threshold_meta": threshold_meta,
        "data_protocol": {
            "training": "healthy_train_windows_only",
            "threshold_calibration": "healthy_validation_scores_only",
            "evaluation": "healthy_test_windows_plus_fault_test_windows",
            "fault_data_used_for_training_or_threshold": False,
        },
        "data_paths": {
            "train_healthy": context.dataset.paths.train_healthy.as_posix(),
            "val_healthy": context.dataset.paths.val_healthy.as_posix(),
            "test_healthy": context.dataset.paths.test_healthy.as_posix(),
            "test_fault": context.dataset.paths.test_fault.as_posix(),
            "fault_labels": context.dataset.paths.fault_labels.as_posix(),
            "preprocessing_config": (context.args.metadata_root / "preprocessing_config.json").as_posix(),
        },
        "data_shapes": {
            "train_healthy": context.dataset.train_shape,
            "val_healthy": context.dataset.val_shape,
            "test_healthy": context.dataset.test_healthy_shape,
            "test_fault": context.dataset.test_fault_shape,
            "fault_labels": context.dataset.fault_labels_shape,
        },
        "device": str(context.device),
        "batch_size": int(context.batch_size),
        "output_files": {
            "metrics_json": (run_dir / "metrics.json").as_posix(),
            "val_healthy_scores_npy": (run_dir / "val_healthy_scores.npy").as_posix(),
            "test_scores_npy": (run_dir / "test_scores.npy").as_posix(),
            "test_labels_npy": (run_dir / "test_labels.npy").as_posix(),
            "run_config_json": (run_dir / "run_config.json").as_posix(),
        },
    }
    save_numpy_array(run_dir / "val_healthy_scores.npy", val_scores, np.float32)
    save_numpy_array(run_dir / "test_scores.npy", test_scores, np.float32)
    save_numpy_array(run_dir / "test_labels.npy", test_labels, np.int64)
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(run_dir / "run_config.json", run_config)
    return {
        "model": model_name,
        "seed": int(seed),
        "threshold_rule": threshold_rule,
        "status": "success",
        "score_source": bundle.score_source,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall_fault": metrics["recall_fault"],
        "far": metrics["far"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "threshold": metrics["threshold"],
        "num_test_healthy": metrics["num_test_healthy"],
        "num_test_fault": metrics["num_test_fault"],
        "num_predicted_anomalies": metrics["num_predicted_anomalies"],
        "num_true_anomalies": metrics["num_true_anomalies"],
        "metrics_path": (run_dir / "metrics.json").as_posix(),
        "run_config_path": (run_dir / "run_config.json").as_posix(),
        "error": "",
    }


def make_error_row(model_name: str, seed: int, threshold_rule: str, error: Exception) -> dict[str, Any]:
    return {
        "model": model_name,
        "seed": int(seed),
        "threshold_rule": threshold_rule,
        "status": "error",
        "score_source": "",
        "auroc": "",
        "auprc": "",
        "f1": "",
        "precision": "",
        "recall_fault": "",
        "far": "",
        "false_alarm_rate": "",
        "threshold": "",
        "num_test_healthy": "",
        "num_test_fault": "",
        "num_predicted_anomalies": "",
        "num_true_anomalies": "",
        "metrics_path": "",
        "run_config_path": "",
        "error": str(error),
    }


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, str) and value.isdigit():
        return value
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def finite_float_values(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric_name, "")
        if value == "" or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def mean_and_sample_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    array = np.asarray(values, dtype=np.float64)
    mean_value = float(array.mean())
    if array.size < 2:
        return mean_value, 0.0
    return mean_value, float(array.std(ddof=1))


def build_summary_by_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_keys = sorted({(str(row.get("model", "")), str(row.get("threshold_rule", ""))) for row in rows})
    output: list[dict[str, Any]] = []
    metric_names = ("auroc", "auprc", "f1", "precision", "recall_fault", "far", "threshold")
    for model_name, threshold_rule in grouped_keys:
        group_rows = [
            row
            for row in rows
            if str(row.get("model", "")) == model_name and str(row.get("threshold_rule", "")) == threshold_rule
        ]
        success_rows = [row for row in group_rows if row.get("status") in {"success", "reused"}]
        error_rows = [row for row in group_rows if row.get("status") == "error"]
        status = "success" if success_rows and not error_rows else "partial" if success_rows else "error"
        summary: dict[str, Any] = {
            "model": model_name,
            "threshold_rule": threshold_rule,
            "status": status,
            "n_success": len(success_rows),
            "n_error": len(error_rows),
            "seeds_success": " ".join(str(row.get("seed", "")) for row in success_rows),
            "seeds_error": " ".join(str(row.get("seed", "")) for row in error_rows),
            "num_test_healthy": success_rows[0].get("num_test_healthy", "") if success_rows else "",
            "num_test_fault": success_rows[0].get("num_test_fault", "") if success_rows else "",
            "error": "; ".join(str(row.get("error", "")) for row in error_rows if row.get("error")),
        }
        for metric_name in metric_names:
            metric_mean, metric_std = mean_and_sample_std(finite_float_values(success_rows, metric_name))
            summary[f"{metric_name}_mean"] = metric_mean
            summary[f"{metric_name}_std"] = metric_std
        output.append(summary)
    return output


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    if not rows:
        lines.append("| " + " | ".join(["(none)"] + [""] * (len(headers) - 1)) + " |")
        return "\n".join(lines)
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def build_summary_md(rows: list[dict[str, Any]], context: RunContext) -> str:
    success_rows = [row for row in rows if row.get("status") in {"success", "reused"}]
    error_rows = [row for row in rows if row.get("status") == "error"]
    table_headers = [
        "model",
        "seed",
        "threshold_rule",
        "status",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall_fault",
        "far",
        "threshold",
    ]
    lines = [
        "# Unified Paderborn Baseline Summary",
        "",
        "## Protocol",
        "- Training data: healthy training windows only.",
        "- Threshold calibration: healthy validation scores only.",
        "- Evaluation data: healthy test windows plus damaged/fault test windows.",
        "- Larger score means more abnormal for every model.",
        "",
        "## Data Reused",
        f"- Processed root: `{context.args.processed_root.as_posix()}`",
        f"- Metadata root: `{context.args.metadata_root.as_posix()}`",
        f"- Train healthy shape: `{context.dataset.train_shape}`",
        f"- Val healthy shape: `{context.dataset.val_shape}`",
        f"- Test healthy shape: `{context.dataset.test_healthy_shape}`",
        f"- Test fault shape: `{context.dataset.test_fault_shape}`",
        "",
        "## Results",
        markdown_table(table_headers, success_rows),
        "",
        "## Errors",
    ]
    if error_rows:
        for row in error_rows:
            lines.append(
                f"- `{row['model']}` seed `{row['seed']}` rule `{row['threshold_rule']}`: {row['error']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Reproducibility",
            f"- Command: `{context.command}`",
            f"- Started at: `{context.started_at}`",
            f"- Device: `{context.device}`",
            f"- Batch size: `{context.batch_size}`",
            "",
        ]
    )
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_")


def build_latex_table(rows: list[dict[str, Any]]) -> str:
    success_rows = [row for row in rows if row.get("status") in {"success", "reused"}]
    lines = [
        "\\begin{tabular}{lllrrrrrr}",
        "\\hline",
        "Model & Seed & Threshold & AUROC & AUPRC & F1 & Precision & Recall & FAR \\\\",
        "\\hline",
    ]
    for row in success_rows:
        lines.append(
            f"{latex_escape(row['model'])} & {row['seed']} & {latex_escape(row['threshold_rule'])} & "
            f"{format_float(row['auroc'])} & {format_float(row['auprc'])} & {format_float(row['f1'])} & "
            f"{format_float(row['precision'])} & {format_float(row['recall_fault'])} & {format_float(row['far'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def format_mean_std(row: dict[str, Any], metric_name: str) -> str:
    mean_value = row.get(f"{metric_name}_mean", "")
    std_value = row.get(f"{metric_name}_std", "")
    if mean_value == "":
        return ""
    if row.get("n_success") == 1:
        return format_float(mean_value)
    return f"{format_float(mean_value)} $\\pm$ {format_float(std_value)}"


def build_latex_table_by_model(rows: list[dict[str, Any]]) -> str:
    summary_rows = [row for row in build_summary_by_model_rows(rows) if int(row.get("n_success", 0)) > 0]
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\hline",
        "Model & Threshold & Seeds & AUROC & AUPRC & F1 & Recall & FAR \\\\",
        "\\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['threshold_rule'])} & {row['n_success']} & "
            f"{format_mean_std(row, 'auroc')} & {format_mean_std(row, 'auprc')} & "
            f"{format_mean_std(row, 'f1')} & {format_mean_std(row, 'recall_fault')} & "
            f"{format_mean_std(row, 'far')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_summaries(output_root: Path, rows: list[dict[str, Any]], context: RunContext) -> None:
    summary_by_model_rows = build_summary_by_model_rows(rows)
    write_csv(output_root / "summary.csv", rows, SUMMARY_FIELDNAMES)
    write_csv(output_root / "summary_by_model.csv", summary_by_model_rows, SUMMARY_BY_MODEL_FIELDNAMES)
    write_text(output_root / "summary.md", build_summary_md(rows, context))
    write_text(output_root / "latex_table.tex", build_latex_table(rows))
    write_text(output_root / "latex_table_by_model.tex", build_latex_table_by_model(rows))


def bool_status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def check_successful_run(row: dict[str, Any]) -> dict[str, Any]:
    model_name = str(row.get("model", ""))
    seed = str(row.get("seed", ""))
    threshold_rule = str(row.get("threshold_rule", ""))
    check_row: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "threshold_rule": threshold_rule,
        "status": "PASS",
        "threshold_from_val": "FAIL",
        "labels_binary": "FAIL",
        "far_matches": "FAIL",
        "no_fault_threshold": "FAIL",
        "notes": "",
    }
    try:
        metrics_path = Path(str(row["metrics_path"]))
        run_dir = metrics_path.parent
        metrics = read_json(metrics_path)
        run_config = read_json(Path(str(row["run_config_path"])))
        val_scores = np.load(run_dir / "val_healthy_scores.npy")
        test_scores = np.load(run_dir / "test_scores.npy")
        test_labels = np.load(run_dir / "test_labels.npy")

        threshold_meta = calibrate_threshold(np.asarray(val_scores, dtype=np.float32), threshold_rule)
        threshold_delta = abs(float(threshold_meta["threshold"]) - float(metrics["threshold"]))
        threshold_ok = threshold_delta <= 1e-6

        unique_labels = set(int(value) for value in np.unique(test_labels).tolist())
        labels_ok = unique_labels == {0, 1}

        healthy_mask = np.asarray(test_labels) == 0
        far_recomputed = float(((np.asarray(test_scores) >= float(metrics["threshold"])) & healthy_mask).sum() / healthy_mask.sum())
        far_delta = abs(far_recomputed - float(metrics.get("far", metrics.get("false_alarm_rate"))))
        far_ok = far_delta <= 1e-12

        run_threshold_meta = run_config.get("threshold_meta", {})
        data_protocol = run_config.get("data_protocol", {})
        no_fault_threshold_ok = (
            run_threshold_meta.get("fit_split") == "val_healthy"
            and int(run_threshold_meta.get("fit_count", -1)) == int(np.asarray(val_scores).reshape(-1).shape[0])
            and data_protocol.get("threshold_calibration") == "healthy_validation_scores_only"
            and data_protocol.get("fault_data_used_for_training_or_threshold") is False
        )

        check_row.update(
            {
                "threshold_from_val": bool_status(threshold_ok),
                "labels_binary": bool_status(labels_ok),
                "far_matches": bool_status(far_ok),
                "no_fault_threshold": bool_status(no_fault_threshold_ok),
                "status": bool_status(threshold_ok and labels_ok and far_ok and no_fault_threshold_ok),
                "notes": (
                    f"threshold_delta={threshold_delta:.3e}; far_delta={far_delta:.3e}; "
                    f"labels={sorted(unique_labels)}"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report validation problems without crashing the run
        check_row["status"] = "FAIL"
        check_row["notes"] = str(exc)
    return check_row


def build_reproduction_check_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    check_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in {"success", "reused"}:
            check_rows.append(
                {
                    "model": row.get("model", ""),
                    "seed": row.get("seed", ""),
                    "threshold_rule": row.get("threshold_rule", ""),
                    "status": "SKIP",
                    "threshold_from_val": "SKIP",
                    "labels_binary": "SKIP",
                    "far_matches": "SKIP",
                    "no_fault_threshold": "SKIP",
                    "notes": row.get("error", "run failed"),
                }
            )
            continue
        check_rows.append(check_successful_run(row))
    return check_rows


def build_reproduction_check_md(rows: list[dict[str, Any]], context: RunContext) -> str:
    check_rows = build_reproduction_check_rows(rows)
    hard_failures = [row for row in check_rows if row["status"] == "FAIL"]
    headers = [
        "model",
        "seed",
        "threshold_rule",
        "status",
        "threshold_from_val",
        "labels_binary",
        "far_matches",
        "no_fault_threshold",
        "notes",
    ]
    lines = [
        "# Paderborn Unified Baseline Reproduction Check",
        "",
        "## Checks",
        "- Thresholds are recomputed from each run's `val_healthy_scores.npy` and compared with `metrics.json`.",
        "- `test_labels.npy` must contain both `0` healthy and `1` fault labels.",
        "- FAR is recomputed only over healthy test labels and compared with `metrics.json`.",
        "- `run_config.json` must record validation-only threshold calibration and no fault use for training or thresholding.",
        "",
        "## Overall Status",
        f"- {'PASS' if not hard_failures else 'FAIL'}",
        "",
        "## Per-Run Checks",
        markdown_table(headers, check_rows),
        "",
        "## Data Protocol",
        f"- Train healthy path: `{context.dataset.paths.train_healthy.as_posix()}`",
        f"- Validation healthy path: `{context.dataset.paths.val_healthy.as_posix()}`",
        f"- Test healthy path: `{context.dataset.paths.test_healthy.as_posix()}`",
        f"- Test fault path: `{context.dataset.paths.test_fault.as_posix()}`",
        "",
    ]
    return "\n".join(lines)


def write_reproduction_check(output_root: Path, rows: list[dict[str, Any]], context: RunContext) -> None:
    write_text(output_root / "reproduction_check.md", build_reproduction_check_md(rows, context))


def build_report_md(rows: list[dict[str, Any]], context: RunContext) -> str:
    success_rows = [row for row in rows if row.get("status") in {"success", "reused"}]
    error_rows = [row for row in rows if row.get("status") == "error"]
    successful_models = sorted({str(row["model"]) for row in success_rows})
    attempted_models = {str(row["model"]) for row in rows}
    not_run_models = sorted(set(DEFAULT_MODELS) - attempted_models)
    python_executable = Path(sys.executable).as_posix()
    shallow_smoke_command = format_command(
        [
            python_executable,
            "scripts/eval_paderborn_baselines_unified.py",
            "--models",
            "isolation_forest",
            "--threshold-rule",
            "percentile_99_5",
            "--seed",
            "42",
            "--skip-train-if-artifacts-exist",
        ]
    )
    neural_smoke_command = format_command(
        [
            python_executable,
            "scripts/eval_paderborn_baselines_unified.py",
            "--models",
            "resdilated_ae",
            "--threshold-rule",
            "percentile_99_5",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--skip-train-if-artifacts-exist",
        ]
    )
    combined_refresh_command = format_command(
        [
            python_executable,
            "scripts/eval_paderborn_baselines_unified.py",
            "--models",
            "isolation_forest",
            "resdilated_ae",
            "--threshold-rule",
            "percentile_99_5",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--skip-train-if-artifacts-exist",
        ]
    )
    lines = [
        "# Paderborn Unified Baseline Runner",
        "",
        "## What Was Implemented",
        "- Added `scripts/eval_paderborn_baselines_unified.py` as a shared healthy-only runner for Paderborn baselines.",
        "- Added common data discovery, deterministic seeding, scoring, validation-only threshold calibration, metrics, and reporting.",
        "- Implemented `ocsvm`, `isolation_forest`, `compact_ae`, `resdilated_ae`, `conv_vae`, and `deep_svdd` through a model registry pattern.",
        "- ConvVAE reuses existing `scripts/train_generative_upgrades.py` model and checkpoint-scoring helpers.",
        "- Deep SVDD uses a compact 1D-CNN encoder, fixed healthy-train center, and squared distance-to-center anomaly scores.",
        "",
        "## Existing Data And Project Paths Reused",
        f"- `train_healthy`: `{context.dataset.paths.train_healthy.as_posix()}`",
        f"- `val_healthy`: `{context.dataset.paths.val_healthy.as_posix()}`",
        f"- `test_healthy`: `{context.dataset.paths.test_healthy.as_posix()}`",
        f"- `test_fault`: `{context.dataset.paths.test_fault.as_posix()}`",
        f"- `fault_labels`: `{context.dataset.paths.fault_labels.as_posix()}`",
        f"- `preprocessing_config`: `{(context.args.metadata_root / 'preprocessing_config.json').as_posix()}`",
        "",
        "## Models Ran Successfully",
        "- " + (", ".join(successful_models) if successful_models else "none"),
        "",
        "## Models Not Run In Latest Smoke Summary",
        "- " + (", ".join(not_run_models) if not_run_models else "none"),
        "",
        "## Models Needing Checkpoints Or Further Work",
    ]
    if error_rows:
        for row in error_rows:
            lines.append(f"- `{row['model']}` seed `{row['seed']}`: {row['error']}")
    else:
        lines.append("- none in the latest run")
    lines.extend(
        [
            "",
            "## Smoke Test Commands",
            f"- Shallow smoke test: `{shallow_smoke_command}`",
            f"- Neural smoke test: `{neural_smoke_command}`",
            f"- Combined summary refresh: `{combined_refresh_command}`",
            "",
            "## Latest Command Observed By Runner",
            f"- `{context.command}`",
            "",
            "## Output Artifacts",
            f"- Summary CSV: `{(context.args.output_root / 'summary.csv').as_posix()}`",
            f"- Summary by model CSV: `{(context.args.output_root / 'summary_by_model.csv').as_posix()}`",
            f"- Summary MD: `{(context.args.output_root / 'summary.md').as_posix()}`",
            f"- LaTeX table: `{(context.args.output_root / 'latex_table.tex').as_posix()}`",
            f"- LaTeX by model table: `{(context.args.output_root / 'latex_table_by_model.tex').as_posix()}`",
            f"- Reproduction check: `{(context.args.output_root / 'reproduction_check.md').as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], context: RunContext) -> None:
    write_text(
        PROJECT_ROOT / "reports" / "paderborn" / "baselines" / "paderborn_unified_baseline_runner.md",
        build_report_md(rows, context),
    )


def run_model_seed(
    *,
    context: RunContext,
    model_name: str,
    seed: int,
    threshold_rules: list[str],
    feature_cache: dict[str, Any],
    registry: dict[str, Callable[[RunContext, int, dict[str, Any]], ScoreBundle]],
) -> list[dict[str, Any]]:
    if (
        not context.args.force
        and context.args.skip_train_if_artifacts_exist
        and all_threshold_outputs_complete(context.args.output_root, model_name, seed, threshold_rules)
    ):
        log(f"Reusing complete unified artifacts for {model_name} seed {seed}")
        return load_existing_rows(context.args.output_root, model_name, seed, threshold_rules)

    if (
        not context.args.force
        and all_threshold_outputs_complete(context.args.output_root, model_name, seed, threshold_rules)
    ):
        log(f"Unified artifacts already exist for {model_name} seed {seed}; use --force to recompute")
        return load_existing_rows(context.args.output_root, model_name, seed, threshold_rules)

    try:
        set_global_seed(seed)
        log(f"Running {model_name} seed {seed}")
        bundle = registry[model_name](context, seed, feature_cache)
        validate_score_bundle(bundle, context.dataset)
        return [
            save_threshold_artifacts(
                context=context,
                model_name=model_name,
                seed=seed,
                threshold_rule=rule,
                bundle=bundle,
            )
            for rule in threshold_rules
        ]
    except Exception as exc:  # noqa: BLE001 - per-model failure should not stop the full run
        log(f"ERROR for {model_name} seed {seed}: {exc}")
        return [make_error_row(model_name, seed, rule, exc) for rule in threshold_rules]


def main() -> int:
    args = parse_args()
    args.processed_root = args.processed_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    models = expand_models(args.models)
    threshold_rules = expand_threshold_rules(args.threshold_rule)
    seeds = expand_seeds(args)
    dataset = discover_dataset(args.processed_root, args.metadata_root)
    device = select_device(args.device)
    batch_size = effective_batch_size(args, device)
    command = format_command([Path(sys.executable).as_posix(), *sys.argv])
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    context = RunContext(
        args=args,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        command=command,
        started_at=started_at,
    )

    registry: dict[str, Callable[[RunContext, int, dict[str, Any]], ScoreBundle]] = {
        "ocsvm": run_ocsvm,
        "isolation_forest": run_isolation_forest,
        "compact_ae": run_compact_ae,
        "resdilated_ae": run_resdilated_ae,
        "conv_vae": run_conv_vae,
        "deep_svdd": run_deep_svdd,
    }
    feature_cache: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    log(f"Models: {', '.join(models)}")
    log(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    log(f"Threshold rules: {', '.join(threshold_rules)}")
    log(f"Output root: {args.output_root.as_posix()}")

    for model_name in models:
        for seed in seeds:
            summary_rows.extend(
                run_model_seed(
                    context=context,
                    model_name=model_name,
                    seed=seed,
                    threshold_rules=threshold_rules,
                    feature_cache=feature_cache,
                    registry=registry,
                )
            )

    write_summaries(args.output_root, summary_rows, context)
    write_reproduction_check(args.output_root, summary_rows, context)
    write_report(summary_rows, context)
    success_count = sum(1 for row in summary_rows if row.get("status") in {"success", "reused"})
    error_count = sum(1 for row in summary_rows if row.get("status") == "error")
    log(f"Summary rows: {len(summary_rows)}; successful/reused: {success_count}; errors: {error_count}")
    log(f"Wrote {args.output_root / 'summary.csv'}")
    return 0 if success_count > 0 or error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

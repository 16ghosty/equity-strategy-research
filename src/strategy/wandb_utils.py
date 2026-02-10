"""
Utility helpers for optional Weights & Biases logging.

The integration is intentionally optional: if wandb is not installed or disabled,
all helpers no-op and the rest of the pipeline remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import subprocess
import time

import pandas as pd


@dataclass
class WandbSettings:
    """Runtime settings for optional wandb logging."""
    enabled: bool = False
    project: str = "equity-strategy"
    entity: Optional[str] = None
    mode: Optional[str] = None  # "online", "offline", or "disabled"/None
    group: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_artifacts: bool = True
    auto_sync: bool = True
    sync_clean: bool = False
    sync_clean_force: bool = True
    sync_path: Optional[str] = "wandb"
    sync_retries: int = 3
    sync_retry_delay_seconds: float = 2.0
    sync_timeout_seconds: int = 180


def _import_wandb():
    """Import wandb lazily so the package remains optional."""
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def is_wandb_available() -> bool:
    """Return True if wandb can be imported."""
    return _import_wandb() is not None


def init_wandb_run(
    settings: WandbSettings,
    config: Optional[dict[str, Any]] = None,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Any:
    """
    Initialize a wandb run or return None if disabled/unavailable.
    """
    if not settings.enabled:
        return None

    wandb = _import_wandb()
    if wandb is None:
        return None

    try:
        run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            mode=settings.mode,
            group=group if group is not None else settings.group,
            name=run_name if run_name is not None else settings.run_name,
            tags=tags if tags is not None else settings.tags,
            notes=settings.notes,
            config=config or {},
            reinit="finish_previous",
        )
        return run
    except Exception:
        return None


def log_metrics(run: Any, metrics: dict[str, Any], prefix: Optional[str] = None) -> None:
    """Log metrics dict to an active wandb run."""
    if run is None:
        return
    payload = metrics
    if prefix:
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
    run.log(payload)


def set_summary_metrics(run: Any, metrics: dict[str, Any], prefix: Optional[str] = None) -> None:
    """Write metrics directly into W&B run summary key-values."""
    if run is None:
        return
    payload = metrics
    if prefix:
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
    try:
        for k, v in payload.items():
            run.summary[k] = v
    except Exception:
        pass


def log_dataframe_table(run: Any, name: str, df: pd.DataFrame) -> None:
    """Log DataFrame as a wandb table."""
    if run is None or df is None or df.empty:
        return

    wandb = _import_wandb()
    if wandb is None:
        return

    table = wandb.Table(dataframe=df)
    run.log({name: table})


def log_artifact_dir(
    run: Any,
    settings: WandbSettings,
    directory: Path,
    artifact_name: str,
    artifact_type: str = "run-output",
) -> None:
    """
    Log an entire directory as an artifact.
    """
    if run is None or not settings.log_artifacts:
        return
    if not directory.exists():
        return

    wandb = _import_wandb()
    if wandb is None:
        return

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)


def log_inline_images(
    run: Any,
    directory: Path,
    key_prefix: str = "charts",
    patterns: tuple[str, ...] = ("*.png",),
    recursive: bool = True,
    max_images: Optional[int] = None,
) -> int:
    """
    Log image files from a directory as inline W&B images.

    Returns number of images logged.
    """
    if run is None:
        return 0
    if not directory.exists():
        return 0

    wandb = _import_wandb()
    if wandb is None:
        return 0

    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(directory.rglob(pattern))
        else:
            files.extend(directory.glob(pattern))

    # Deduplicate while preserving deterministic order.
    unique_files = sorted(set(files))
    logged = 0

    for image_path in unique_files:
        if max_images is not None and logged >= max_images:
            break
        if not image_path.is_file():
            continue

        rel = image_path.relative_to(directory).as_posix()
        key = f"{key_prefix}/{rel.replace('/', '__').replace('.', '_')}"
        run.log({key: wandb.Image(str(image_path), caption=rel)})
        logged += 1

    return logged


def log_plotly_figures(
    run: Any,
    figures: dict[str, Any],
    key_prefix: str = "interactive",
) -> int:
    """
    Log Plotly figures to W&B for interactive charting.

    Returns number of figures logged.
    """
    if run is None or not figures:
        return 0

    logged = 0
    for name, fig in figures.items():
        if fig is None:
            continue
        run.log({f"{key_prefix}/{name}": fig})
        logged += 1
    return logged


def _run_sync_command(cmd: list[str], timeout_seconds: int) -> bool:
    """Execute wandb sync command and return True on zero exit code."""
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception:
        return False
    return completed.returncode == 0


def sync_runs(settings: WandbSettings, extra_paths: Optional[list[str]] = None) -> bool:
    """
    Sync offline/saved wandb runs to the cloud.

    Returns:
        True when sync command executes without subprocess errors.
    """
    if not settings.enabled or settings.mode == "disabled":
        return False

    sync_paths: list[str] = []
    if extra_paths:
        sync_paths.extend(str(p) for p in extra_paths if p)
    if settings.sync_path:
        sync_paths.append(str(settings.sync_path))
    if not sync_paths:
        sync_paths = ["wandb"]

    # Preserve order while dropping duplicates.
    sync_paths = list(dict.fromkeys(sync_paths))

    base_cmd = [
        "wandb",
        "sync",
        "--sync-all",
        "--include-offline",
        "--no-include-synced",
        "--mark-synced",
    ]
    if settings.project:
        base_cmd.extend(["--project", settings.project])
    if settings.entity:
        base_cmd.extend(["--entity", settings.entity])
    if settings.sync_clean:
        base_cmd.append("--clean")
        if settings.sync_clean_force:
            base_cmd.append("--clean-force")

    max_attempts = max(1, int(settings.sync_retries))
    for sync_path in sync_paths:
        cmd = [*base_cmd, sync_path]
        for attempt in range(max_attempts):
            if _run_sync_command(cmd, timeout_seconds=settings.sync_timeout_seconds):
                return True
            if attempt < max_attempts - 1:
                time.sleep(max(0.0, float(settings.sync_retry_delay_seconds)))
    return False


def finish_run(run: Any, settings: Optional[WandbSettings] = None) -> None:
    """Finish an active wandb run and optionally auto-sync local runs."""
    sync_paths: list[str] = []
    if run is not None:
        try:
            run_dir = getattr(run, "dir", None)
            if run_dir:
                run_path = Path(run_dir)
                # W&B run.dir typically points to a "files" directory.
                if run_path.name == "files":
                    run_path = run_path.parent
                sync_paths.append(str(run_path))
        except Exception:
            pass

        try:
            run.finish()
        except Exception:
            pass

    if settings is not None and settings.auto_sync:
        sync_runs(settings, extra_paths=sync_paths)

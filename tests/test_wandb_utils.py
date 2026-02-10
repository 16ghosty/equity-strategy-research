"""
Unit tests for wandb utility helpers.
"""

from types import SimpleNamespace

from strategy.wandb_utils import WandbSettings, finish_run, sync_runs


def test_sync_runs_retries_until_success(monkeypatch):
    """sync_runs should retry failed sync attempts."""
    calls = {"count": 0}

    def _fake_run(*args, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(returncode=0 if calls["count"] >= 2 else 1)

    monkeypatch.setattr("strategy.wandb_utils.subprocess.run", _fake_run)
    monkeypatch.setattr("strategy.wandb_utils.time.sleep", lambda *_: None)

    settings = WandbSettings(
        enabled=True,
        mode="online",
        sync_retries=3,
        sync_retry_delay_seconds=0.0,
    )

    assert sync_runs(settings) is True
    assert calls["count"] == 2


def test_finish_run_attempts_sync_even_without_run(monkeypatch):
    """finish_run should still trigger sync when run is None."""
    called = {"value": False}

    def _fake_sync(settings, extra_paths=None):
        called["value"] = True
        return True

    monkeypatch.setattr("strategy.wandb_utils.sync_runs", _fake_sync)

    finish_run(
        run=None,
        settings=WandbSettings(enabled=True, auto_sync=True, mode="online"),
    )

    assert called["value"] is True


def test_finish_run_passes_run_directory_to_sync(monkeypatch):
    """finish_run should use run.dir parent path for targeted syncing."""
    captured = {"paths": None}

    class _FakeRun:
        dir = "/tmp/wandb/offline-run-abc/files"

        def finish(self):
            return None

    def _fake_sync(settings, extra_paths=None):
        captured["paths"] = extra_paths
        return True

    monkeypatch.setattr("strategy.wandb_utils.sync_runs", _fake_sync)

    finish_run(
        run=_FakeRun(),
        settings=WandbSettings(enabled=True, auto_sync=True, mode="online"),
    )

    assert captured["paths"] == ["/tmp/wandb/offline-run-abc"]

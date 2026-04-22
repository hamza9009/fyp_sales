"""Console progress tracking helpers for the ML training pipeline."""

from __future__ import annotations

from typing import Any

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ImportError:  # pragma: no cover - graceful fallback when rich is unavailable
    Progress = None


class TrainingProgressTracker:
    """Optional Rich progress bar for long-running training sessions."""

    def __init__(
        self,
        *,
        total_steps: int,
        enabled: bool = False,
        initial_description: str = "Starting ML training pipeline",
    ) -> None:
        self._enabled = bool(enabled and Progress is not None)
        self._total_steps = total_steps
        self._initial_description = initial_description
        self._progress: Any | None = None
        self._task_id: Any | None = None

    def __enter__(self) -> "TrainingProgressTracker":
        if self._enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self._initial_description,
                total=self._total_steps,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is not None:
            self._progress.stop()

    def advance(self, description: str) -> None:
        """Advance the visible progress bar by one step."""
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1, description=description)

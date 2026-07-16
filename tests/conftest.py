"""pytest configuration for FolderHistory."""

from __future__ import annotations

import hypothesis
from _pytest.config import Config

# ── Hypothesis profile ──────────────────────────────────────────────────────

hypothesis.settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,
    suppress_health_check=list(hypothesis.HealthCheck),
)
hypothesis.settings.register_profile(
    "dev",
    max_examples=100,
    deadline=None,
)
hypothesis.settings.load_profile("dev")


def pytest_configure(config: Config) -> None:
    """Register custom markers and configure test infrastructure."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: marks benchmark tests (run separately with '-m benchmark')",
    )

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ValidationCheckResult:
    name: str
    status: str
    details: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    plugin_name: str
    dataset_path: str | None
    generated_at: str
    checks: list[ValidationCheckResult]
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        *,
        plugin_name: str,
        dataset_path: str | None,
        checks: list[ValidationCheckResult],
        metrics: dict[str, Any] | None = None,
    ) -> "ValidationReport":
        timestamp = datetime.now(timezone.utc).isoformat()
        return cls(
            plugin_name=plugin_name,
            dataset_path=dataset_path,
            generated_at=timestamp,
            checks=checks,
            metrics=metrics or {},
        )

    @property
    def passed(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def summary(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
        for check in self.checks:
            if check.status in counts:
                counts[check.status] += 1
        counts["total"] = len(self.checks)
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin_name,
            "dataset_path": self.dataset_path,
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "metrics": self.metrics,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "details": check.details,
                    "metrics": check.metrics,
                }
                for check in self.checks
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_text(self) -> str:
        summary = self.summary()
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Plugin: {self.plugin_name}",
            f"Dataset: {self.dataset_path or 'unknown'}",
            f"Status: {status} (passes={summary['pass']}, fails={summary['fail']}, "
            f"warnings={summary['warn']}, skipped={summary['skip']})",
            "Checks:",
        ]
        for check in self.checks:
            detail = f" - {check.details}" if check.details else ""
            lines.append(f"- {check.name}: {check.status.upper()}{detail}")
            if check.metrics:
                lines.append(f"  metrics={check.metrics}")
        if self.metrics:
            lines.append("Metrics:")
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

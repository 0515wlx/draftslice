"""Release manifests and report generation.

A release run is a set of versioned result files plus a manifest that records
the file hashes, code commit, and environment. The report renderer is
deliberately data-driven: it accepts every implemented benchmark schema and
emits one Markdown table section per schema, so adding a result type does not
require hand-editing a release document.

This module is not a scheduler or a runner. It never invents numbers: an empty
or malformed result is rejected, and missing metrics are reported as ``n/a``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RELEASE_SCHEMA = "release/1"
KNOWN_SCHEMAS = {
    "bench/1": "head_accuracy",
    "baseline_matrix/1": "baseline_matrix",
    "factorial/1": "factorial",
    "constraint_bench/1": "constraint_plan",
    "kernel_bench/1": "kernel",
    "e2e/1": "end_to_end",
}


class ReleaseError(ValueError):
    """Raised when a release manifest or result is malformed."""


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_result(value: Any, *, source: str = "result") -> str:
    """Return the schema id or raise ``ReleaseError``."""

    if not isinstance(value, dict):
        raise ReleaseError(f"{source}: result must be a JSON object")
    schema = value.get("schema_version")
    if schema not in KNOWN_SCHEMAS:
        raise ReleaseError(f"{source}: unknown or missing schema_version {schema!r}")
    if schema == "bench/1":
        if "metrics" not in value or "name" not in value:
            raise ReleaseError(f"{source}: bench/1 needs name and metrics")
    elif schema == "baseline_matrix/1":
        if "policies" not in value or not isinstance(value["policies"], dict):
            raise ReleaseError(f"{source}: baseline_matrix/1 needs policies")
    elif schema == "factorial/1":
        if "cells" not in value or not isinstance(value["cells"], list):
            raise ReleaseError(f"{source}: factorial/1 needs cells")
    elif schema == "constraint_bench/1":
        if "summary" not in value:
            raise ReleaseError(f"{source}: constraint_bench/1 needs summary")
    elif schema == "kernel_bench/1":
        if "records" not in value and "summary" not in value:
            raise ReleaseError(f"{source}: kernel_bench/1 needs records or summary")
    elif schema == "e2e/1":
        if "methods" not in value and "summary" not in value:
            raise ReleaseError(f"{source}: e2e/1 needs methods or summary")
    return str(schema)


def load_result(path: str | Path) -> tuple[dict[str, Any], str]:
    source = str(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value, validate_result(value, source=source)


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    path: str
    sha256: str
    schema: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema": self.schema,
            "kind": self.kind,
        }


@dataclass(slots=True)
class ReleaseManifest:
    name: str
    files: list[ReleaseFile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    git_commit: str = ""
    git_dirty: bool = False
    env: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RELEASE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "env": dict(self.env),
            "metadata": dict(self.metadata),
            "files": [item.to_dict() for item in self.files],
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ReleaseManifest:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Any) -> ReleaseManifest:
        if not isinstance(value, dict):
            raise ReleaseError("manifest must be an object")
        if value.get("schema_version") != RELEASE_SCHEMA:
            raise ReleaseError(
                f"unsupported manifest schema {value.get('schema_version')!r}"
            )
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ReleaseError("manifest needs a non-empty name")
        files = []
        for raw in value.get("files", []):
            if not isinstance(raw, dict):
                raise ReleaseError("manifest files must be objects")
            try:
                files.append(
                    ReleaseFile(
                        path=str(raw["path"]),
                        sha256=str(raw["sha256"]),
                        schema=str(raw["schema"]),
                        kind=str(raw["kind"]),
                    )
                )
            except KeyError as exc:
                raise ReleaseError(f"manifest file is missing {exc}") from exc
        git = value.get("git", {})
        return cls(
            name=name,
            files=files,
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", _utc_now())),
            git_commit=str(git.get("commit", "")),
            git_dirty=bool(git.get("dirty", False)),
            env=dict(value.get("env", {})),
        )


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def _git_state(root: str | Path | None = None) -> tuple[str, bool]:
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:  # pragma: no cover - git is optional at report time
        return "", False


def build_manifest(
    name: str,
    inputs: list[str | Path],
    *,
    metadata: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> ReleaseManifest:
    """Hash every result file and build a release manifest."""

    if not name:
        raise ReleaseError("release name must not be empty")
    if not inputs:
        raise ReleaseError("a release needs at least one result file")
    files = []
    for path in inputs:
        _, schema = load_result(path)
        files.append(
            ReleaseFile(
                path=str(path),
                sha256=file_hash(path),
                schema=schema,
                kind=KNOWN_SCHEMAS[schema],
            )
        )
    commit, dirty = _git_state(root)
    return ReleaseManifest(
        name=name,
        files=files,
        metadata=dict(metadata or {}),
        git_commit=commit,
        git_dirty=dirty,
        env=_environment(),
    )


def verify_manifest(
    manifest: ReleaseManifest,
    *,
    check_hashes: bool = True,
) -> list[str]:
    """Return human-readable problems instead of silently accepting a release."""

    problems: list[str] = []
    for item in manifest.files:
        path = Path(item.path)
        if not path.exists():
            problems.append(f"missing file: {item.path}")
            continue
        if check_hashes and file_hash(path) != item.sha256:
            problems.append(f"hash mismatch: {item.path}")
            continue
        try:
            load_result(path)
        except (ReleaseError, json.JSONDecodeError) as exc:
            problems.append(str(exc))
    return problems


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return lines


def render_report(manifest: ReleaseManifest, results: list[dict[str, Any]]) -> str:
    """Render one Markdown report from validated results."""

    if len(results) != len(manifest.files):
        raise ReleaseError("results and manifest files have different lengths")
    lines = [f"# Release report: {manifest.name}", ""]
    lines.append(f"- Created: `{manifest.created_at}`")
    lines.append(f"- Commit: `{manifest.git_commit or 'unknown'}`")
    lines.append(f"- Git dirty: `{manifest.git_dirty}`")
    lines.append(f"- Files: {len(manifest.files)}")
    for key, value in sorted(manifest.metadata.items()):
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    for item, value in zip(manifest.files, results, strict=True):
        lines.append(f"## {item.kind}: `{Path(item.path).name}`")
        lines.append("")
        lines.append(f"- Schema: `{item.schema}`")
        lines.append(f"- SHA-256: `{item.sha256}`")
        lines.append("")
        schema = item.schema
        if schema == "bench/1":
            metrics = value.get("metrics", {})
            rows = [[key, metrics[key]] for key in sorted(metrics)]
            lines.extend(_table(["metric", "value"], rows))
        elif schema == "baseline_matrix/1":
            rows = []
            for policy, result in sorted(value.get("policies", {}).items()):
                metrics = result.get("metrics", {})
                rows.append(
                    [
                        policy,
                        metrics.get("top1"),
                        metrics.get("set_recall@5", metrics.get("set_recall")),
                        metrics.get("prefix_coverage@5"),
                    ]
                )
            lines.extend(
                _table(["policy", "top1", "set_recall@5", "prefix_coverage@5"], rows)
            )
        elif schema == "factorial/1":
            rows = []
            for cell in value.get("cells", []):
                rows.append(
                    [
                        cell.get("screener"),
                        cell.get("case"),
                        cell.get("coverage"),
                        cell.get("mean_candidates"),
                        cell.get("candidate_bytes"),
                    ]
                )
            lines.extend(
                _table(
                    ["screener", "case", "coverage", "mean_candidates", "bytes"],
                    rows,
                )
            )
        elif schema == "constraint_bench/1":
            summary = value.get("summary", {})
            rows = [[key, summary[key]] for key in sorted(summary)]
            lines.extend(_table(["summary", "value"], rows))
        elif schema == "kernel_bench/1":
            records = value.get("records", [])
            rows = []
            for record in records:
                rows.append(
                    [
                        record.get("name"),
                        record.get("batch"),
                        record.get("rows"),
                        record.get("median_us"),
                        record.get("effective_bandwidth_gbps"),
                    ]
                )
            if rows:
                lines.extend(
                    _table(
                        ["kernel", "batch", "rows", "median_us", "eff_gbps"],
                        rows,
                    )
                )
            else:
                summary = value.get("summary", {})
                rows = [[key, summary[key]] for key in sorted(summary)]
                lines.extend(_table(["summary", "value"], rows))
        elif schema == "e2e/1":
            methods = value.get("methods")
            if methods is None:
                methods = {value.get("method", "method"): value}
            rows = []
            for method, result in sorted(methods.items()):
                summary = result.get("summary", {})
                rows.append(
                    [
                        method,
                        summary.get("records"),
                        summary.get("validity"),
                        summary.get("tau"),
                        summary.get("tokens_per_s"),
                    ]
                )
            lines.extend(
                _table(["method", "records", "validity", "tau", "tok/s"], rows)
            )
        warnings = value.get("warnings")
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_release(
    name: str,
    inputs: list[str | Path],
    *,
    out: str | Path,
    metadata: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> tuple[ReleaseManifest, str]:
    """Validate inputs, write a manifest beside ``out``, and return the report."""

    manifest = build_manifest(name, inputs, metadata=metadata, root=root)
    problems = verify_manifest(manifest)
    if problems:
        raise ReleaseError("; ".join(problems))
    results = [load_result(path)[0] for path in inputs]
    report = render_report(manifest, results)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    manifest.write(out_path.with_suffix(".manifest.json"))
    return manifest, report


def compare_manifests(
    base: ReleaseManifest,
    candidate: ReleaseManifest,
) -> str:
    """Render a B5 regression table between two releases.

    Files are matched by their result kind. Missing kinds are reported, not
    silently skipped; a candidate run that dropped a workload is a regression in
    coverage even when its remaining numbers improve.
    """

    base_kinds = [item.kind for item in base.files]
    candidate_kinds = [item.kind for item in candidate.files]
    lines = [f"# B5 comparison: {base.name} -> {candidate.name}", ""]
    missing = sorted(set(base_kinds) - set(candidate_kinds))
    added = sorted(set(candidate_kinds) - set(base_kinds))
    if missing:
        lines.append(f"- Missing in candidate: {', '.join(missing)}")
    if added:
        lines.append(f"- Added in candidate: {', '.join(added)}")
    if not missing and not added:
        lines.append("- Result kinds match.")
    lines.append("")
    lines.extend(
        _table(
            ["release", "created_at", "commit", "dirty", "files"],
            [
                [base.name, base.created_at, base.git_commit, base.git_dirty, len(base.files)],
                [
                    candidate.name,
                    candidate.created_at,
                    candidate.git_commit,
                    candidate.git_dirty,
                    len(candidate.files),
                ],
            ],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "KNOWN_SCHEMAS",
    "RELEASE_SCHEMA",
    "ReleaseError",
    "ReleaseFile",
    "ReleaseManifest",
    "build_manifest",
    "build_release",
    "compare_manifests",
    "file_hash",
    "load_result",
    "render_report",
    "validate_result",
    "verify_manifest",
]

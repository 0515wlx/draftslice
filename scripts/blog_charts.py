#!/usr/bin/env python3
"""Generate the two dependency-free SVG figures used by the blog draft.

Data sources (gitignored local workspace, see ``runs/``):

- ``runs/p2p4_verify_20261123/chat_ab_tool_b1.json`` — chat-shape tau.
- ``runs/p2p4_verify_20261123/confirm_4b_tool_b1_k8_20261123.json`` — raw-prompt
  tau and the 1.19x speedup.
- ``runs/P2_P4_HEADROOM_20261123.md`` — measured head share and the head-only
  ceiling per k/B (copied into the constants below because the trace is a
  profiler dump, not a small result JSON).

Run from the repo root::

    python scripts/blog_charts.py

Writes ``docs/blog/tau-workload-shape.svg`` and
``docs/blog/head-share-vs-k.svg``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = (
    ROOT / "runs" / "p2p4_verify_20261123",
    ROOT / "scripts" / "runs" / "p2p4_verify_20261123",
)
RUNS = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])
OUT = ROOT / "docs" / "blog"


def _load(name: str) -> dict:
    path = RUNS / name
    if not path.exists():
        raise SystemExit(f"missing {path}; run from a workspace that has runs/")
    return json.loads(path.read_text(encoding="utf-8"))


def _bar_chart(
    groups: list[tuple[str, float, float, str]],
    *,
    y_max: float,
    y_label: str,
    series: tuple[str, str],
    title: str,
    colors: tuple[str, str] = ("#9aa5b1", "#2563eb"),
) -> str:
    """One grouped bar per workload; returns an SVG string."""

    width, height = 720, 400
    left, right, top, bottom = 64, 24, 56, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot = plot_w / len(groups)
    bar_w = slot * 0.28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter,system-ui,sans-serif">',
        f'<text x="{left}" y="30" font-size="17" font-weight="600" fill="#111827">{title}</text>',
        f'<text x="{left}" y="48" font-size="11" fill="#6b7280">{series[0]} vs {series[1]}</text>',
    ]
    # Axes and grid.
    for step in range(6):
        value = y_max * step / 5
        y = top + plot_h - plot_h * value / y_max
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="10" fill="#6b7280" '
            f'text-anchor="end">{value:.1f}</text>'
        )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" font-size="11" fill="#374151" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle">{y_label}</text>'
    )
    for index, (label, low, high, note) in enumerate(groups):
        center = left + slot * (index + 0.5)
        for offset, value, color in (
            (-bar_w * 0.55, low, colors[0]),
            (bar_w * 0.55, high, colors[1]),
        ):
            bar_h = plot_h * value / y_max
            x = center + offset - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" font-size="11" '
                f'fill="#111827" text-anchor="middle">{value:.2f}</text>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 22}" font-size="12" '
            f'fill="#111827" text-anchor="middle">{label}</text>'
        )
        if note:
            parts.append(
                f'<text x="{center:.1f}" y="{top + plot_h + 40}" font-size="11" '
                f'fill="#6b7280" text-anchor="middle">{note}</text>'
            )
    # Legend.
    for index, (name, color) in enumerate(zip(series, colors, strict=False)):
        x = left + plot_w - 190 + index * 100
        parts.append(
            f'<rect x="{x}" y="20" width="10" height="10" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 15}" y="29" font-size="11" fill="#374151">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    chat = _load("chat_ab_tool_b1.json")["summary"]
    raw = _load("confirm_4b_tool_b1_k8_20261123.json")["summary"]["labels"]["slice"]
    # The raw-prompt MTP tau is the paired baseline for the 1.19x result. The
    # report records the slice tau (7.67) and the ratio 2.25; recover mtp from
    # the ratio so both figures come from the same run.
    raw_mtp = round(raw["tau"] / raw["median_tau_ratio"], 2)
    chat_mtp = round(chat["mtp_tau"], 2)
    chat_slice = round(chat["slice_tau"], 2)
    raw_slice = round(raw["tau"], 2)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tau-workload-shape.svg").write_text(
        _bar_chart(
            [
                ("raw completion", raw_mtp, raw_slice, "1.19x median"),
                ("chat template", chat_mtp, chat_slice, "1.00x median"),
            ],
            y_max=8.0,
            y_label="accept length tau",
            series=("built-in MTP", "DraftSlice slice"),
            title="Acceptance depends on the request shape, not the kernel",
        ),
        encoding="utf-8",
    )
    (OUT / "head-share-vs-k.svg").write_text(
        _bar_chart(
            [
                ("k=4, B=1", 9.5, 9.4, "head 3.6 ms/step"),
                ("k=8, B=1", 16.2, 17.3, "head 7.8 ms/step"),
                ("k=16, B=1", 21.8, 24.8, "head 16.6 ms/step"),
                ("k=8, B=8", 15.2, 15.8, "head 8.9 ms/step"),
            ],
            y_max=27.0,
            y_label="percent of step",
            series=("draft-head share", "max gain if rows free"),
            title="The draft head is a real, batch-invariant budget",
            colors=("#f59e0b", "#059669"),
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'tau-workload-shape.svg'}")
    print(f"wrote {OUT / 'head-share-vs-k.svg'}")


if __name__ == "__main__":
    main()

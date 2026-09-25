#!/usr/bin/env python3
"""Render BEIR quality scores from the local PG18 quality matrix."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = (
    ROOT
    / 'docs/performance/data/diagnostics/'
    / 'pg18-beir-quality-matrix-current-2026-05-06.json'
)
OUTPUT_SVG = (
    ROOT
    / 'docs/performance/reports/'
    / 'pg18-quality-score-heatmap-2026-05-06.svg'
)

WIDTH = 1900
HEIGHT = 1380
LEFT = 64
RIGHT = 64
TOP = 250
PANEL_GAP_X = 70
PANEL_GAP_Y = 96

PANEL_W = (WIDTH - LEFT - RIGHT - PANEL_GAP_X) / 2
PANEL_H = 450
LABEL_W = 180
DOC_W = 72
HEADER_H = 58
ROW_H = 23
BEST_EPSILON = 0.001
GRID_STROKE = '#e2d6c4'
PANEL_STROKE = '#d8ccb8'

METRICS = [
    ('ndcg@10', 'NDCG@10'),
    ('map@100', 'MAP@100'),
    ('recall@100', 'Recall@100'),
    ('precision@10', 'Precision@10'),
]

ENGINES = [
    ('upstream_bm25s', 'bm25s'),
    ('evoke_ids', 'evoke ids'),
    ('evoke_text', 'evoke text[]'),
    ('pg_search', 'pg_search'),
    ('vchord_bm25', 'vchord_bm25'),
]


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def human_count(value: int) -> str:
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if value >= 1_000:
        return f'{value / 1_000:.0f}K'
    return str(value)


def svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 14,
    weight: str = '400',
    fill: str = '#25211a',
    anchor: str = 'middle',
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">'
        f'{escape(text)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = 'none',
    stroke_width: float = 1.0,
    rx: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" rx="{rx:.2f}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = '#cfc4b2',
    stroke_width: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
    )


def lerp(start: int, end: int, amount: float) -> int:
    return round(start + (end - start) * amount)


def blend(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> str:
    amount = max(0.0, min(1.0, amount))
    r = lerp(start[0], end[0], amount)
    g = lerp(start[1], end[1], amount)
    b = lerp(start[2], end[2], amount)
    return f'#{r:02x}{g:02x}{b:02x}'


def score_color(
    score: float,
    min_score: float,
    max_score: float,
) -> tuple[str, str]:
    if max_score <= min_score:
        intensity = 0.0
    else:
        intensity = (score - min_score) / (max_score - min_score)
    neutral = (248, 243, 233)
    accent = (24, 100, 171)
    fill = blend(neutral, accent, intensity)
    text = '#ffffff' if intensity >= 0.64 else '#1f1b16'
    return fill, text


def load_rows() -> list[dict[str, Any]]:
    with INPUT_JSON.open(encoding='utf-8') as fh:
        data = json.load(fh)

    rows: list[dict[str, Any]] = []
    for dataset, payload in data['results'].items():
        if 'error' in payload:
            continue
        stats = payload['stats']
        metrics_by_engine: dict[str, dict[str, float]] = {}
        for key, _ in ENGINES:
            engine_payload = payload.get(key)
            if not isinstance(engine_payload, dict):
                continue
            metrics = engine_payload.get('metrics')
            if isinstance(metrics, dict):
                metrics_by_engine[key] = {
                    metric: float(metrics[metric])
                    for metric, _ in METRICS
                    if metric in metrics
                }
        rows.append(
            {
                'dataset': dataset,
                'documents': int(stats['documents']),
                'metrics': metrics_by_engine,
            }
        )

    rows.sort(key=lambda row: row['documents'])
    return rows


def metric_bounds(
    rows: list[dict[str, Any]],
    metric_key: str,
) -> tuple[float, float]:
    values = [
        row['metrics'][engine_key][metric_key]
        for row in rows
        for engine_key, _ in ENGINES
        if engine_key in row['metrics']
    ]
    return min(values), max(values)


def draw_legend(
    parts: list[str],
    y: float,
) -> None:
    add = parts.append
    bar_x = WIDTH / 2 - 260
    bar_y = y
    step_count = 28
    step_w = 520 / step_count
    for idx in range(step_count):
        fraction = idx / (step_count - 1)
        fill, _ = score_color(fraction, 0.0, 1.0)
        add(rect(bar_x + idx * step_w, bar_y, step_w + 0.5, 18, fill=fill))
    add(rect(bar_x, bar_y, 520, 18, fill='none', stroke=GRID_STROKE))
    add(svg_text(bar_x, bar_y + 39, 'panel min', size=12,
                 fill='#6b6257'))
    add(svg_text(bar_x + 260, bar_y + 39, 'absolute cell values', size=12,
                 fill='#6b6257'))
    add(svg_text(bar_x + 520, bar_y + 39, 'panel max', size=12,
                 fill='#6b6257'))
    add(
        svg_text(
            WIDTH / 2,
            bar_y - 14,
            'color is scaled independently inside each metric panel',
            size=13,
            weight='600',
            fill='#51483d',
        )
    )
    add(svg_text(bar_x - 18, bar_y + 14, 'lower', size=12,
                 fill='#74695d', anchor='end'))
    add(svg_text(bar_x + 538, bar_y + 14, 'higher', size=12,
                 fill='#244f7e', anchor='start'))


def draw_panel(
    parts: list[str],
    rows: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
    panel_x: float,
    panel_y: float,
) -> None:
    add = parts.append
    min_score, max_score = metric_bounds(rows, metric_key)
    add(rect(panel_x, panel_y, PANEL_W, PANEL_H, fill='#fffaf0',
             stroke=PANEL_STROKE, stroke_width=1.2, rx=14))
    add(svg_text(panel_x + PANEL_W / 2, panel_y - 20, metric_label,
                 size=20, weight='700'))

    table_x = panel_x + 18
    table_y = panel_y + 22
    table_w = PANEL_W - 36
    engine_area_w = table_w - LABEL_W - DOC_W
    cell_w = engine_area_w / len(ENGINES)

    add(svg_text(table_x, table_y + 24, 'Dataset', size=12,
                 weight='700', fill='#5d5348', anchor='start'))
    add(svg_text(table_x + LABEL_W + DOC_W / 2, table_y + 24, 'Docs',
                 size=12, weight='700', fill='#5d5348'))
    for idx, (_, label) in enumerate(ENGINES):
        x = table_x + LABEL_W + DOC_W + idx * cell_w + cell_w / 2
        add(svg_text(x, table_y + 18, label, size=11, weight='700',
                     fill='#5d5348'))

    header_bottom = table_y + HEADER_H
    add(line(table_x, header_bottom, table_x + table_w, header_bottom,
             stroke=GRID_STROKE, stroke_width=1.2))

    for row_idx, row in enumerate(rows):
        y = header_bottom + row_idx * ROW_H
        if row_idx % 2 == 0:
            add(rect(table_x, y, table_w, ROW_H, fill='#fffdf8'))
        dataset = row['dataset']
        docs = human_count(row['documents'])
        add(svg_text(table_x, y + 16, dataset, size=11,
                     fill='#383229', anchor='start'))
        add(svg_text(table_x + LABEL_W + DOC_W / 2, y + 16, docs,
                     size=11, fill='#6c6258'))

        row_scores = [
            row['metrics'][engine_key][metric_key]
            for engine_key, _ in ENGINES
            if engine_key in row['metrics']
        ]
        best_score = max(row_scores)
        for engine_idx, (engine_key, _) in enumerate(ENGINES):
            x = table_x + LABEL_W + DOC_W + engine_idx * cell_w
            metric = row['metrics'][engine_key][metric_key]
            fill, text_fill = score_color(metric, min_score, max_score)
            is_near_best = best_score - metric <= BEST_EPSILON
            add(
                rect(
                    x,
                    y,
                    cell_w,
                    ROW_H,
                    fill=fill,
                    stroke=GRID_STROKE,
                    stroke_width=1.0,
                )
            )
            weight = '700' if is_near_best else '400'
            add(
                svg_text(
                    x + cell_w / 2,
                    y + 16,
                    f'{metric:.4f}',
                    size=10,
                    fill=text_fill,
                    weight=weight,
                )
            )

    add(line(table_x + LABEL_W, table_y, table_x + LABEL_W,
             header_bottom + len(rows) * ROW_H, stroke=GRID_STROKE))
    add(line(table_x + LABEL_W + DOC_W, table_y,
             table_x + LABEL_W + DOC_W,
             header_bottom + len(rows) * ROW_H, stroke=GRID_STROKE))


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit('No successful quality rows found')

    parts: list[str] = []
    add = parts.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    add('<rect width="100%" height="100%" fill="#fffdf8"/>')
    add(svg_text(WIDTH / 2, 50, 'BEIR Quality Score Heatmap',
                 size=32, weight='800'))
    add(
        svg_text(
            WIDTH / 2,
            80,
            'qrels-only evaluation | top100 retrieval | '
            'cell = absolute metric score',
            size=15,
            fill='#5b5349',
        )
    )
    add(
        svg_text(
            WIDTH / 2,
            108,
            'Datasets are sorted by document count. '
            f'Bold cells are within {BEST_EPSILON:.3f} of the '
            'best engine for that dataset and metric.',
            size=14,
            fill='#74695d',
        )
    )

    draw_legend(parts, 154)

    panel_positions = [
        (LEFT, TOP),
        (LEFT + PANEL_W + PANEL_GAP_X, TOP),
        (LEFT, TOP + PANEL_H + PANEL_GAP_Y),
        (LEFT + PANEL_W + PANEL_GAP_X, TOP + PANEL_H + PANEL_GAP_Y),
    ]
    for (metric_key, metric_label), (panel_x, panel_y) in zip(
        METRICS,
        panel_positions,
        strict=True,
    ):
        draw_panel(parts, rows, metric_key, metric_label, panel_x, panel_y)

    add(
        svg_text(
            WIDTH / 2,
            HEIGHT - 38,
            'This chart compares relevance quality only. '
            'Use the QPS, build-time, and index-size matrices for '
            'engineering cost trade-offs.',
            size=14,
            fill='#6a6258',
        )
    )
    add('</svg>')

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    print(OUTPUT_SVG)


if __name__ == '__main__':
    main()

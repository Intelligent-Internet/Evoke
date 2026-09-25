#!/usr/bin/env python3
"""Render a build-time-vs-dataset-scale SVG chart from the PG18 matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = (
    ROOT
    / 'docs/performance/data/diagnostics/'
    / 'pg18-beir-extension-matrix-current-2026-04-02.json'
)
OUTPUT_SVG = (
    ROOT
    / 'docs/performance/reports/'
    / 'pg18-build-vs-dataset-scale-2026-04-02.svg'
)

WIDTH = 1600
HEIGHT = 1240
LEFT = 120
RIGHT = 60
TOP = 90
BOTTOM = 500
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM

ENGINES = [
    ('upstream_bm25s', 'upstream bm25s', '#1f77b4', 'circle'),
    ('evoke_ids', 'evoke ids', '#d62728', 'triangle_up'),
    ('evoke_text', 'evoke text[]', '#ff7f0e', 'triangle_down'),
    ('pg_search', 'pg_search', '#2ca02c', 'square'),
    ('vchord_bm25', 'vchord_bm25', '#9467bd', 'diamond'),
]


def human_count(value: int) -> str:
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if value >= 1_000:
        return f'{value / 1_000:.0f}K'
    return str(value)


def human_tick(value: float) -> str:
    rounded = int(round(value))
    if rounded >= 1_000_000:
        if rounded % 1_000_000 == 0:
            return f'{rounded // 1_000_000}.0M'
        return f'{rounded / 1_000_000:.1f}M'
    if rounded >= 1_000:
        return f'{rounded / 1_000:.0f}K'
    return str(rounded)


def human_time(value_s: float) -> str:
    if value_s < 1:
        return f'{round(value_s * 1000):.0f}ms'
    if value_s < 60:
        return f'{value_s:.0f}s'
    if value_s < 3600:
        return f'{value_s / 60:.0f}m'
    return f'{value_s / 3600:.1f}h'


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    weight: str = '400',
    fill: str = '#222',
    anchor: str = 'middle',
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">{text}</text>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = '#bbb',
    width: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"/>'
    )


def svg_polyline(
    points: list[tuple[float, float]],
    *,
    stroke: str = '#bbb',
    width: float = 1.0,
) -> str:
    pts = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width:.2f}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def marker_svg(x: float, y: float, kind: str, color: str) -> str:
    if kind == 'circle':
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" '
            f'fill="{color}" stroke="white" stroke-width="1.5"/>'
        )
    if kind == 'square':
        size = 10
        return (
            f'<rect x="{x - size / 2:.2f}" y="{y - size / 2:.2f}" '
            f'width="{size}" height="{size}" fill="{color}" '
            f'stroke="white" stroke-width="1.5"/>'
        )
    if kind == 'diamond':
        points = [
            (x, y - 6.5),
            (x + 6.5, y),
            (x, y + 6.5),
            (x - 6.5, y),
        ]
    elif kind == 'triangle_up':
        points = [
            (x, y - 7),
            (x + 7, y + 6),
            (x - 7, y + 6),
        ]
    elif kind == 'triangle_down':
        points = [
            (x, y + 7),
            (x + 7, y - 6),
            (x - 7, y - 6),
        ]
    else:
        raise ValueError(f'unknown marker: {kind}')
    pts = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
    return (
        f'<polygon points="{pts}" fill="{color}" '
        f'stroke="white" stroke-width="1.5"/>'
    )


def load_rows() -> list[dict[str, object]]:
    with INPUT_JSON.open() as fh:
        data = json.load(fh)
    rows = []
    for dataset, payload in data['results'].items():
        row = {
            'dataset': dataset,
            'documents': payload['stats']['documents'],
            'queries': payload['stats']['queries'],
            'series': {},
        }
        for key, _, _, _ in ENGINES:
            entry = payload[key]
            row['series'][key] = {
                'qps': float(entry['query']['qps']),
                'build_s': float(entry['build_ms']) / 1000.0,
            }
        rows.append(row)
    rows.sort(key=lambda row: row['documents'])
    return rows


def log_scale(value: float, minimum: float, maximum: float, span: float) -> float:
    return (
        math.log10(value) - math.log10(minimum)
    ) / (
        math.log10(maximum) - math.log10(minimum)
    ) * span


def main() -> None:
    rows = load_rows()
    min_docs = min(row['documents'] for row in rows)
    max_docs = max(row['documents'] for row in rows)
    min_build = min(
        row['series'][engine]['build_s']
        for row in rows
        for engine, _, _, _ in ENGINES
    )
    max_build = max(
        row['series'][engine]['build_s']
        for row in rows
        for engine, _, _, _ in ENGINES
    )

    x_min = min_docs
    x_max = 10 ** math.ceil(math.log10(max_docs))
    y_min = 10 ** math.floor(math.log10(min_build))
    y_max = 10 ** (math.ceil(math.log10(max_build)) + 1)

    parts: list[str] = []
    add = parts.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    add('<rect width="100%" height="100%" fill="#fffdf8"/>')
    add(
        svg_text(
            WIDTH / 2,
            42,
            'Index Build Time vs Dataset Scale (PG18 BEIR Matrix)',
            size=28,
            weight='700',
        )
    )
    add(
        svg_text(
            WIDTH / 2,
            68,
            'x-axis: document count (log scale) | '
            'y-axis: build time (log scale)',
            size=15,
            fill='#555',
        )
    )

    start_power = math.ceil(math.log10(x_min))
    for power in range(int(start_power), int(math.log10(x_max)) + 1):
        value = 10 ** power
        x = LEFT + log_scale(value, x_min, x_max, PLOT_W)
        add(svg_line(x, TOP, x, TOP + PLOT_H, stroke='#e8e2d6', width=1))
        add(svg_text(x, TOP + PLOT_H + 24, human_tick(value), size=13,
                     fill='#666'))

    first_tick_x = LEFT + log_scale(x_min, x_min, x_max, PLOT_W)
    add(svg_line(first_tick_x, TOP, first_tick_x, TOP + PLOT_H,
                 stroke='#e8e2d6', width=1))
    add(svg_text(first_tick_x, TOP + PLOT_H + 24, human_tick(x_min),
                 size=13, fill='#666'))

    for power in range(int(math.log10(y_min)), int(math.log10(y_max)) + 1):
        value = 10 ** power
        y = TOP + PLOT_H - log_scale(value, y_min, y_max, PLOT_H)
        add(svg_line(LEFT, y, LEFT + PLOT_W, y, stroke='#e8e2d6', width=1))
        add(svg_text(LEFT - 16, y + 4, human_time(value), size=13,
                     fill='#666', anchor='end'))

    add(svg_line(LEFT, TOP, LEFT, TOP + PLOT_H, stroke='#444', width=1.5))
    add(svg_line(LEFT, TOP + PLOT_H, LEFT + PLOT_W, TOP + PLOT_H,
                 stroke='#444', width=1.5))

    for key, _label, color, marker in ENGINES:
        points: list[tuple[float, float]] = []
        for row in rows:
            x = LEFT + log_scale(row['documents'], x_min, x_max, PLOT_W)
            y = TOP + PLOT_H - log_scale(
                row['series'][key]['build_s'], y_min, y_max, PLOT_H
            )
            points.append((x, y))
        path = ' '.join(
            [
                f'M {points[0][0]:.2f} {points[0][1]:.2f}',
                *[f'L {x:.2f} {y:.2f}' for x, y in points[1:]],
            ]
        )
        add(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
        for x, y in points:
            add(marker_svg(x, y, marker, color))

    label_rows = [
        TOP + PLOT_H + 56,
        TOP + PLOT_H + 106,
        TOP + PLOT_H + 156,
        TOP + PLOT_H + 206,
        TOP + PLOT_H + 256,
        TOP + PLOT_H + 306,
    ]
    row_ends = [-10_000.0 for _ in label_rows]
    label_layouts = []
    for row in rows:
        x = LEFT + log_scale(row['documents'], x_min, x_max, PLOT_W)
        docs_label = f'{human_count(row["documents"])} docs'
        half_width = max(
            len(row['dataset']) * 3.9,
            len(docs_label) * 3.5,
        ) + 12
        x_lo = LEFT + half_width + 6
        x_hi = LEFT + PLOT_W - half_width - 6
        best = None
        for lane_idx, lane_y in enumerate(label_rows):
            label_x = min(max(x, x_lo), x_hi)
            min_start = row_ends[lane_idx] + 16
            start = label_x - half_width
            if start < min_start:
                label_x += min_start - start
            if label_x > x_hi:
                continue
            cost = abs(label_x - x) + lane_idx * 8
            if best is None or cost < best[0]:
                best = (cost, lane_idx, lane_y, label_x, docs_label, half_width)
        if best is None:
            lane_idx = len(label_rows) - 1
            lane_y = label_rows[lane_idx]
            label_x = x_hi
            docs_label = f'{human_count(row["documents"])} docs'
            half_width = max(
                len(row['dataset']) * 3.9,
                len(docs_label) * 3.5,
            ) + 12
        else:
            _, lane_idx, lane_y, label_x, docs_label, half_width = best
        row_ends[lane_idx] = label_x + half_width
        label_layouts.append(
            {
                'dataset': row['dataset'],
                'point_x': x,
                'label_x': label_x,
                'label_y': lane_y,
                'docs_label': docs_label,
            }
        )

    for layout in label_layouts:
        x = layout['point_x']
        label_x = layout['label_x']
        y_base = TOP + PLOT_H
        y_label = layout['label_y']
        add(
            svg_polyline(
                [
                    (x, y_base + 2),
                    (x, y_base + 16),
                    (label_x, y_label - 28),
                ],
                stroke='#c8c1b4',
                width=1,
            )
        )
        add(
            svg_text(
                label_x,
                y_label,
                layout['dataset'],
                size=13,
                weight='600',
            )
        )
        add(
            svg_text(
                label_x,
                y_label + 18,
                layout['docs_label'],
                size=12,
                fill='#666',
            )
        )

    add(svg_text(LEFT + PLOT_W / 2, HEIGHT - 18, 'Dataset size (documents)',
                 size=16, weight='600'))
    add(
        f'<g transform="translate(28 {TOP + PLOT_H / 2}) rotate(-90)">'
        f'{svg_text(0, 0, "Index build time", size=16, weight="600")}'
        f'</g>'
    )

    legend_x = LEFT + PLOT_W - 230
    legend_y = TOP + 24
    add(
        f'<rect x="{legend_x - 16}" y="{legend_y - 28}" width="242" '
        f'height="168" rx="10" fill="#fffaf0" stroke="#d8cfbf"/>'
    )
    add(svg_text(legend_x + 95, legend_y - 8, 'Engines', size=16,
                 weight='700'))
    for idx, (_key, label, color, marker) in enumerate(ENGINES):
        y = legend_y + idx * 28
        add(svg_line(legend_x, y, legend_x + 26, y, stroke=color, width=3))
        add(marker_svg(legend_x + 13, y, marker, color))
        add(svg_text(legend_x + 38, y + 5, label, size=14,
                     fill='#333', anchor='start'))

    add(
        svg_text(
            LEFT,
            HEIGHT - 84,
            'Note: build time reflects index construction only; it excludes '
            'query execution and per-VM orchestration overhead.',
            size=13,
            fill='#666',
            anchor='start',
        )
    )
    add(
        svg_text(
            LEFT,
            HEIGHT - 58,
            'Representative largest dataset: msmarco | upstream 3.2m | '
            'ids 56.06s | text[] 1.5m | pg_search 1.2m | vchord 1.7m',
            size=13,
            fill='#666',
            anchor='start',
        )
    )
    add('</svg>')

    OUTPUT_SVG.write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()

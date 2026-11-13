"""Matplotlib drawing utilities for environment rendering."""

from typing import Iterable, Sequence

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from actions.reward_parts import ActedAircraftRewardRow, RewardParts


def draw_bounds(ax, xy_min: float, xy_max: float) -> None:
    """Draw a rectangular boundary box from (xy_min, xy_min) to (xy_max, xy_max)."""

    ax.plot(
        [xy_min, xy_max, xy_max, xy_min, xy_min],
        [xy_min, xy_min, xy_max, xy_max, xy_min],
        linestyle="-",
        color="black",
        linewidth=0.8,
        alpha=0.6,
    )


def draw_sector_polygons(
    ax,
    polygons: Sequence[Sequence[tuple[float, float]]],
    colors: Sequence,
    *,
    alpha: float = 0.3,
) -> None:
    """Draw sector polygons as fill-only overlays without overlap darkening."""

    if not polygons:
        return
    fallback_fill_color = (0.5, 0.5, 0.5, 1.0)
    color_count = len(colors)
    bg_rgba = ax.get_facecolor()
    bg_rgb = np.asarray(bg_rgba[:3], dtype=float)
    for idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        facecolor = colors[idx % color_count] if color_count else fallback_fill_color
        face_rgb = np.asarray(facecolor[:3], dtype=float)
        blended_rgb = alpha * face_rgb + (1.0 - alpha) * bg_rgb
        patch = Polygon(
            np.asarray(polygon, dtype=float),
            closed=True,
            facecolor=(*blended_rgb.tolist(), 1.0),
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
            zorder=0,
            clip_on=True,
        )
        ax.add_patch(patch)


def _compute_dy_axes(ax, fontsize_pts: float = 8.0, linespacing: float = 1.2) -> float:
    """Compute the vertical offset between lines in Axes coordinates."""

    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    dpi = ax.figure.dpi if ax.figure is not None else 100.0
    line_step_px = (fontsize_pts * dpi / 72.0) * linespacing
    return (line_step_px / axes_bbox.height) if axes_bbox is not None else 0.018


def _draw_right_aligned_pair(
    ax,
    y_axes: float,
    label_text: str,
    value_text: str,
    value_color,
    base_color,
) -> None:
    """Draw a right-aligned pair of text labels and values."""

    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    t_val = ax.text(
        0.98,
        y_axes,
        value_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        color=value_color,
        clip_on=True,
    )
    bbox_val = t_val.get_window_extent(renderer=renderer)
    dx_val_axes = bbox_val.width / axes_bbox.width
    ax.text(
        0.98 - dx_val_axes,
        y_axes,
        label_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        color=base_color,
        clip_on=True,
    )


def _draw_right_aligned_segments(
    ax,
    x_right_axes: float,
    y_axes: float,
    segments: list[tuple[str, tuple[float, float, float, float]]],
    *,
    fontsize: float = 8,
    family: str = "monospace",
) -> float:
    """Draw multiple right-aligned text segments with individual colors.

    Segments are drawn from right to left, each as (text, color).
    Returns the x position to the left of the leftmost drawn segment in Axes coords.
    """

    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    x_right = x_right_axes
    for text, color in segments:
        t = ax.text(
            x_right,
            y_axes,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
            family=family,
            color=color,
            clip_on=True,
        )
        bbox = t.get_window_extent(renderer=renderer)
        dx_axes = bbox.width / axes_bbox.width
        x_right = x_right - dx_axes
    return x_right


def draw_route(
    ax,
    route_points: Sequence[tuple[float, float]],
    color,
    xy_min: float | None = None,
    xy_max: float | None = None,
    *,
    linewidth: float = 1.0,
    waypoint_size: float = 12.0,
    waypoint_alpha: float = 0.8,
    include_line: bool = True,
    include_waypoints: bool = True,
) -> None:
    """Draw a route with optional polyline and waypoint markers."""

    if len(route_points) < 2:
        return
    if include_line:
        xs, ys = zip(*route_points)
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9, clip_on=True)
    if include_waypoints:
        draw_waypoints(
            ax,
            route_points,
            color,
            xy_min=xy_min,
            xy_max=xy_max,
            waypoint_size=waypoint_size,
            waypoint_alpha=waypoint_alpha,
        )


def draw_waypoints(
    ax,
    points: Sequence[tuple[float, float]],
    color,
    xy_min: float | None = None,
    xy_max: float | None = None,
    *,
    waypoint_size: float = 12.0,
    waypoint_alpha: float = 0.8,
) -> None:
    """Scatter waypoint markers only; does not draw connecting lines."""

    if not points:
        return
    if xy_min is not None and xy_max is not None:
        pts = [p for p in points if within_bounds(p[0], p[1], xy_min, xy_max)]
    else:
        pts = list(points)
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(
        xs,
        ys,
        marker="o",
        s=waypoint_size,
        c=[color],
        alpha=waypoint_alpha,
        edgecolors="black",
        linewidths=0.4,
        clip_on=True,
    )


def draw_triangle_marker(
    ax,
    x: float,
    y: float,
    color,
    size: float = 60.0,
    xy_min: float | None = None,
    xy_max: float | None = None,
    heading_deg: float | None = None,
) -> None:
    """Draw an oriented, asymmetric triangle at (x, y). Skip if out of bounds."""

    if xy_min is not None and xy_max is not None and not within_bounds(x, y, xy_min, xy_max):
        return

    base = 0.5 * max(1.0, float(np.sqrt(size)))
    length = 0.82 * base
    half_width = 0.30 * base
    tail = 0.50 * base

    # Local triangle: nose at (+length, 0), tail corners at (-tail, ±half_width)
    pts_local = np.array(
        [
            [+length, 0.0],
            [-tail, -half_width],
            [-tail, +half_width],
        ],
        dtype=float,
    )
    # Center the triangle around its centroid so (x, y) is visually centered
    centroid = pts_local.mean(axis=0)
    pts_local = pts_local - centroid
    tri = Polygon(
        pts_local,
        closed=True,
        facecolor=color,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    transform = Affine2D().rotate_deg(float(heading_deg or 0.0)).translate(float(x), float(y))
    tri.set_transform(transform + ax.transData)
    ax.add_patch(tri)


def draw_velocity_line(
    ax,
    x: float,
    y: float,
    vx: float,
    vy: float,
    color,
    target_len: float = 10.0,
    xy_min: float | None = None,
    xy_max: float | None = None,
) -> None:
    """Draw a line from (x, y) in the (vx, vy) direction, scaled to target length."""

    if xy_min is not None and xy_max is not None:
        if not within_bounds(x, y, xy_min, xy_max):
            return
    speed_norm = (vx**2 + vy**2) ** 0.5
    if speed_norm <= 1e-6:
        return
    scale = target_len / speed_norm
    end_x = x + vx * scale
    end_y = y + vy * scale
    # Unified rule: Do not draw if end point is outside bounds
    if xy_min is not None and xy_max is not None:
        if not within_bounds(end_x, end_y, xy_min, xy_max):
            return
    ax.plot([x, end_x], [y, end_y], color=color, linewidth=1.5, alpha=0.9, clip_on=True)


def draw_action_markers(
    ax,
    action_history: Sequence[dict],
    color_map: dict[str, tuple[float, float, float, float]],
    xy_min: float | None = None,
    xy_max: float | None = None,
) -> None:
    """Draw action markers (persist across frames)."""

    if not action_history:
        return
    entries: Iterable[dict] = action_history
    if xy_min is not None and xy_max is not None:
        entries = [e for e in action_history if within_bounds(e["x"], e["y"], xy_min, xy_max)]
    action_x = [e["x"] for e in entries]
    action_y = [e["y"] for e in entries]
    action_colors = [color_map.get(e["call_sign"]) for e in entries]
    ax.scatter(action_x, action_y, marker="x", s=20, c=action_colors, linewidths=0.6, zorder=6)
    for e in entries:
        lx = e["x"] + 1.5
        ly = e["y"] + 1.5
        if xy_min is not None and xy_max is not None and not within_bounds(lx, ly, xy_min, xy_max):
            continue
        ax.text(
            lx,
            ly,
            str(e["action"]),
            fontsize=6,
            color=color_map.get(e["call_sign"]) or (0.2, 0.2, 0.2, 1.0),
            alpha=0.6,
            clip_on=True,
        )


def draw_info_box(
    ax,
    rows: Sequence[str],
    row_colors: Sequence | None = None,
) -> None:
    """Draw a small info box in the lower-right corner with provided rows."""

    if not rows:
        return

    # Draw an invisible multiline text to create the background bbox
    full_text = "\n" + "\n".join(rows)  # Extra newline to match Text spacing
    ax.text(
        0.98,
        0.02,
        full_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        color=(0, 0, 0, 0),  # invisible text
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": (1, 1, 1, 0.8),
            "ec": "gray",
            "lw": 0.8,
        },
        linespacing=1.2,
    )

    # Compute per-line vertical offset in Axes coordinates to match Text spacing
    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    axes_height_px = axes_bbox.height if axes_bbox is not None else None

    fontsize_pts = 8.0
    linespacing = 1.2
    dpi = ax.figure.dpi if ax.figure is not None else 100.0
    # Convert a single line step (in points) -> pixels -> Axes fraction
    line_step_px = (fontsize_pts * dpi / 72.0) * linespacing
    dy_axes = (line_step_px / axes_height_px) if axes_height_px else 0.018  # sane fallback

    # Draw actual colored text lines on top, aligned to the same anchor
    x_ax, y_ax = 0.98, 0.02
    # Draw rows first (bottom to top)
    for i, row_text in enumerate(rows):
        row_color = row_colors[i] if row_colors and i < len(row_colors) else (0, 0, 0, 1)
        ax.text(
            x_ax,
            y_ax,
            row_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            color=row_color,
            linespacing=linespacing,
            clip_on=True,
        )
        y_ax += dy_axes


def draw_reward_overlay(
    ax,
    cumulative_reward: float,
    last_reward: RewardParts,
    acted_rows: Sequence[ActedAircraftRewardRow],
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    """Top-right reward overlay with compact last-step acted aircraft rows."""

    good_color = (0.1, 0.6, 0.1, 0.9)
    bad_color = (0.8, 0.1, 0.1, 0.9)
    base_color = (0.0, 0.0, 0.0, 0.9)

    # First line: cumulative value
    cum_val_color = good_color if float(cumulative_reward) >= 0.0 else bad_color
    cum_val_text = f"{float(cumulative_reward):+5.2f}"
    _draw_right_aligned_pair(
        ax,
        0.98,
        "Cumulative: ",
        cum_val_text,
        cum_val_color,
        base_color,
    )

    dy_axes = _compute_dy_axes(ax)
    y_ax = 0.98 - dy_axes

    def color_for(v: float):
        return good_color if float(v) >= 0.0 else bad_color

    # Second line: overall step total
    step_total_val = float(last_reward.total())
    step_total_color = color_for(step_total_val)
    _draw_right_aligned_pair(
        ax, y_ax, "Step: ", f"{step_total_val:+.3f}", step_total_color, base_color
    )
    y_ax -= dy_axes

    # Then one compact row per acted aircraft in the last step.
    for row in acted_rows:
        agent_fixed = f"{row.agent_id[:7]:<7}"
        agent_color = color_map.get(row.agent_id, base_color)
        segments = [
            (f"#{int(row.action_num):>2} ", base_color),
            (agent_fixed, agent_color),
        ]
        _draw_right_aligned_segments(ax, 0.98, y_ax, segments)
        y_ax -= dy_axes


def draw_world_age_overlay(
    ax,
    world_age_seconds: int,
    extra_lines: Sequence[str] | None = None,
) -> None:
    """Top-left overlay showing world age and optional additional lines."""

    base_color = (0.0, 0.0, 0.0, 0.9)
    lines = [f"t = {world_age_seconds} s"]
    if extra_lines:
        lines.extend(str(line) for line in extra_lines)

    y_axes = 0.98
    dy_axes = _compute_dy_axes(ax)
    for line in lines:
        ax.text(
            0.02,
            y_axes,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            color=base_color,
            clip_on=True,
        )
        y_axes -= dy_axes


# --- helpers ---
def within_bounds(x: float, y: float, xy_min: float, xy_max: float) -> bool:
    return (xy_min <= x <= xy_max) and (xy_min <= y <= xy_max)


class BoundedPlot:
    """Convenience wrapper to draw with fixed bounds without repeating args."""

    def __init__(self, ax, xy_min: float, xy_max: float) -> None:
        self.ax = ax
        self.xy_min = xy_min
        self.xy_max = xy_max

    # Layout helpers
    def set_limits_and_grid(self, title: str | None = None) -> None:
        self.ax.set_xlim(self.xy_min, self.xy_max)
        self.ax.set_ylim(self.xy_min, self.xy_max)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
        self.ax.set_xlabel("X (nm)")
        self.ax.set_ylabel("Y (nm)")
        if title:
            self.ax.set_title(title)

    # Drawing methods using built-ins with bound injection
    def bounds(self) -> None:
        draw_bounds(self.ax, self.xy_min, self.xy_max)

    def sector_polygons(
        self,
        polygons: Sequence[Sequence[tuple[float, float]]],
        colors: Sequence,
        *,
        alpha: float = 0.08,
    ) -> None:
        draw_sector_polygons(self.ax, polygons, colors, alpha=alpha)

    def route(self, route_points: Sequence[tuple[float, float]], color, **kwargs) -> None:
        draw_route(
            self.ax,
            route_points,
            color,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
            **kwargs,
        )

    def triangle(
        self,
        x: float,
        y: float,
        color,
        size: float = 60.0,
        heading_deg: float | None = None,
    ) -> None:
        draw_triangle_marker(
            self.ax,
            x,
            y,
            color,
            size=size,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
            heading_deg=heading_deg,
        )

    def velocity(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        color,
        target_len: float = 10.0,
    ) -> None:
        draw_velocity_line(
            self.ax,
            x,
            y,
            vx,
            vy,
            color,
            target_len=target_len,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
        )

    def actions(self, action_history: Sequence[dict], color_map: dict) -> None:
        draw_action_markers(
            self.ax,
            action_history,
            color_map,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
        )

    def info(
        self,
        rows: Sequence[str],
        row_colors: Sequence | None = None,
    ) -> None:
        draw_info_box(self.ax, rows, row_colors=row_colors)

    def reward(
        self,
        cumulative_reward: float,
        last_parts: RewardParts,
        acted_rows: Sequence[ActedAircraftRewardRow],
        color_map: dict[str, tuple[float, float, float, float]],
    ) -> None:
        draw_reward_overlay(
            self.ax,
            cumulative_reward,
            last_parts,
            acted_rows=acted_rows,
            color_map=color_map,
        )

    def world_age(self, world_age_seconds: int, extra_lines: Sequence[str] | None = None) -> None:
        draw_world_age_overlay(self.ax, world_age_seconds, extra_lines=extra_lines)

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color,
        linewidth: float = 0.5,
        alpha: float = 0.6,
    ) -> None:
        if len(points) < 2:
            return
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, clip_on=True)

    def waypoints(
        self,
        points: Sequence[tuple[float, float]],
        color,
        waypoint_size: float = 12.0,
        waypoint_alpha: float = 0.8,
    ) -> None:
        draw_waypoints(
            self.ax,
            points,
            color,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
            waypoint_size=waypoint_size,
            waypoint_alpha=waypoint_alpha,
        )

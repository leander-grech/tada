"""Matplotlib drawing utilities for environment rendering."""

from typing import Iterable, Sequence

import numpy as np
from config.config import Config
from matplotlib import colormaps
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from utils.reward_utils import RewardParts


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
        pts_local, closed=True, facecolor=color, edgecolor="black", linewidth=0.6, zorder=5
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
    landed_callsigns: set[str] | None = None,
) -> None:
    """Draw action markers (persist across frames).

    Aircraft in ``landed_callsigns`` (already landed) have their markers collapsed to small
    dots with no action number, so their history stops obstructing the action numbers of
    aircraft still being worked. Active aircraft keep the ``x`` marker + action-id label."""

    if not action_history:
        return
    landed_callsigns = landed_callsigns or set()
    entries: Iterable[dict] = action_history
    if xy_min is not None and xy_max is not None:
        entries = [e for e in action_history if within_bounds(e["x"], e["y"], xy_min, xy_max)]

    active = [e for e in entries if e["call_sign"] not in landed_callsigns]
    landed = [e for e in entries if e["call_sign"] in landed_callsigns]

    # Landed aircraft: small dots only (no number).
    if landed:
        ax.scatter(
            [e["x"] for e in landed], [e["y"] for e in landed],
            marker=".", s=10, c=[color_map.get(e["call_sign"]) for e in landed],
            alpha=0.5, linewidths=0.0, zorder=5,
        )

    # Active aircraft: x marker + action-id label.
    if active:
        ax.scatter(
            [e["x"] for e in active], [e["y"] for e in active],
            marker="x", s=26, c=[color_map.get(e["call_sign"]) for e in active],
            linewidths=0.7, zorder=6,
        )
        for e in active:
            lx = e["x"] + 1.5
            ly = e["y"] + 1.5
            if xy_min is not None and xy_max is not None and not within_bounds(lx, ly, xy_min, xy_max):
                continue
            ax.text(
                lx,
                ly,
                str(e["action"]),
                fontsize=8.5,
                fontweight="bold",
                color=color_map.get(e["call_sign"]) or (0.2, 0.2, 0.2, 1.0),
                alpha=0.75,
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
        bbox={"boxstyle": "round,pad=0.3", "fc": (1, 1, 1, 0.8), "ec": "gray", "lw": 0.8},
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


def draw_info_panel(
    ax,
    rows: Sequence[str],
    row_colors: Sequence | None = None,
    summary: str | None = None,
    header: str | None = None,
    footnote: str | None = None,
    fontsize: float = 8.0,
) -> None:
    """Draw the per-aircraft info rows on a dedicated side panel axes (not over the map).

    Rows are listed top-to-bottom in the order given, each in its aircraft colour, under an
    optional ``header`` naming the columns -- the rows are dense fixed-width numbers and are
    unreadable without it. An optional ``summary`` row (e.g. averages) is drawn in bold at the
    bottom. ``header`` must be formatted with the same field widths as the rows (see
    ``render_utils.INFO_HEADER``). The axes is expected to be decoration-free."""

    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.0, 0.995, "AIRCRAFT", ha="left", va="top", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=(0.1, 0.1, 0.1, 1.0))

    # Reserve the bottom band for the summary and the footnote when present.
    bottom = 0.02 + (0.10 if summary else 0.0) + (0.07 if footnote else 0.0)
    top = 0.90
    if header:
        # Column names sit in the row grid's own coordinate system so they align with it, with
        # a rule underneath separating them from the data.
        ax.text(0.0, top, header, ha="left", va="center", transform=ax.transAxes,
                fontsize=fontsize, family="monospace", fontweight="bold",
                color=(0.25, 0.25, 0.25, 1.0))
        ax.axhline(top - 0.028, xmin=0.0, xmax=1.0, color=(0.7, 0.7, 0.7, 1.0), linewidth=0.7)
        top -= 0.075

    if rows:
        n = len(rows)
        span = top - bottom
        for i, row_text in enumerate(rows):
            y = top - span * (i / (n - 1)) if n > 1 else top
            color = row_colors[i] if row_colors and i < len(row_colors) else (0, 0, 0, 1)
            ax.text(0.0, y, row_text, ha="left", va="center", transform=ax.transAxes,
                    fontsize=fontsize, family="monospace", color=color)

    if summary:
        y_summary = 0.02 + (0.07 if footnote else 0.0)
        ax.text(0.0, y_summary, summary, ha="left", va="center", transform=ax.transAxes,
                fontsize=fontsize + 1.0, family="monospace", fontweight="bold",
                color=(0.1, 0.1, 0.1, 1.0))

    if footnote:
        ax.text(0.0, 0.02, footnote, ha="left", va="center", transform=ax.transAxes,
                fontsize=fontsize - 1.0, family="monospace", style="italic",
                color=(0.45, 0.45, 0.45, 1.0))


def draw_airport_3d(
    ax,
    world,
    agent_colors: dict,
    center_xy: tuple[float, float],
    past_tracks: dict | None = None,
    predicted_tracks: dict | None = None,
    zoom_nm: float = 35.0,
    alt_max_kft: float = 22.0,
    trail_len: int = 5,
) -> None:
    """Draw a 3D approach view of the immediate airport vicinity on a 3D axes.

    ``center_xy`` is the airport (marked with a star at ground level) and the view is a
    ``zoom_nm`` half-width box around it, so the descents converge on the star.

    Each under-control aircraft within ``zoom_nm`` of ``center_xy`` is shown as a coloured
    marker at its (x, y, altitude) with a vertical stem to the ground plane. Altitude is in
    thousands of feet (kft). Per aircraft there is also: its predicted trajectory in 3D
    (dotted, the descent profile it is currently headed for), the remaining flight plan on the
    ground plane (dashed), and the last ``trail_len`` points of the flown path (solid)."""

    ax.clear()
    past_tracks = past_tracks or {}
    predicted_tracks = predicted_tracks or {}
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ax.set_xlim(cx - zoom_nm, cx + zoom_nm)
    ax.set_ylim(cy - zoom_nm, cy + zoom_nm)
    ax.set_zlim(0.0, alt_max_kft)

    def _in(px: float, py: float) -> bool:
        return abs(px - cx) <= zoom_nm and abs(py - cy) <= zoom_nm

    # Airport reference at ground level, with a ground ring so the star reads as a place on
    # the plane rather than a floating marker.
    _ring = np.linspace(0.0, 2.0 * np.pi, 60)
    _r = zoom_nm * 0.08
    ax.plot(cx + _r * np.cos(_ring), cy + _r * np.sin(_ring), np.zeros_like(_ring),
            color=(0.35, 0.35, 0.35, 1.0), linewidth=0.7)
    ax.scatter([cx], [cy], [0.0], marker="*", s=140, color="black", depthshade=False, zorder=6)

    for ac in world.aircraft:
        if ac.mission.completed or not ac.under_control:
            continue
        p = ac.state.position_nm_nm_ft
        x, y, z = float(p.x), float(p.y), float(p.z) / 1000.0
        color = agent_colors.get(ac.call_sign, (0.3, 0.3, 0.3, 1.0))

        # Predicted trajectory in 3D: the descent profile this aircraft is headed for if
        # nothing further is commanded. Clipped to the in-view portion so a far-out aircraft
        # still contributes the part of its approach that lands in the box.
        forecast = [
            (fx, fy, fz / 1000.0)
            for (fx, fy, fz) in (predicted_tracks.get(ac.call_sign) or [])
            if _in(fx, fy)
        ]
        if len(forecast) >= 2:
            ax.plot([f[0] for f in forecast], [f[1] for f in forecast],
                    [f[2] for f in forecast],
                    color=color, linewidth=1.1, alpha=0.85, linestyle=(0, (1, 1.6)))

        if not _in(x, y):
            continue

        # Flown path: last `trail_len` in-view track points, solid + transparent (ground plane).
        trail = [(tx, ty) for (tx, ty) in past_tracks.get(ac.call_sign, []) if _in(tx, ty)]
        trail = trail[-trail_len:]
        if len(trail) >= 2:
            ax.plot([t[0] for t in trail], [t[1] for t in trail], [0.0] * len(trail),
                    color=color, linewidth=1.4, alpha=0.35, solid_capstyle="round")

        # Remaining flight plan: route waypoints, dashed (ground plane).
        plan = [(x, y)]
        for rp in (ac.route or []):
            wp = world.waypoints.get(rp.waypoint_name)
            if wp is not None:
                wx, wy = float(wp.position_nm.x), float(wp.position_nm.y)
                if _in(wx, wy):
                    plan.append((wx, wy))
        if len(plan) >= 2:
            ax.plot([q[0] for q in plan], [q[1] for q in plan], [0.0] * len(plan),
                    color=color, linewidth=1.0, alpha=0.75, linestyle=(0, (4, 3)))

        # Altitude stem + aircraft marker.
        ax.plot([x, x], [y, y], [0.0, z], color=color, linewidth=0.8, alpha=0.45)
        ax.scatter([x], [y], [z], color=color, s=28, depthshade=True, edgecolors="black",
                   linewidths=0.3)

    # The unit lives in the title: the z-axis label sits outside the cube on its right, where
    # it lands off the edge of the figure at this layout.
    ax.set_title(f"Approach (3D) — ±{zoom_nm:.0f} nm, altitude in kft",
                 fontsize=11, fontweight="bold", pad=0.0)
    ax.set_xlabel("X (nm)", fontsize=7, labelpad=-9)
    ax.set_ylabel("Y (nm)", fontsize=7, labelpad=-9)
    ax.tick_params(labelsize=6, pad=-3)
    ax.margins(0)
    # Enlarge the cube within its axes to cut the surrounding whitespace.
    try:
        ax.set_box_aspect((1.0, 1.0, 0.62), zoom=1.55)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.view_init(elev=20, azim=-58)


def draw_reward_overlay(
    ax,
    cumulative_reward: float,
    last_reward: RewardParts,
    agent_id: str,
    action_num: int,
    agent_color,
) -> None:
    """Top-right reward overlay with cumulative and last-step components."""

    good_color = (0.1, 0.6, 0.1, 0.9)
    bad_color = (0.8, 0.1, 0.1, 0.9)
    base_color = (0.0, 0.0, 0.0, 0.9)

    agent_fixed = (agent_id or "")[:7]
    agent_fixed = f"{agent_fixed:<7}"

    # First line: cumulative with colored value
    cum_val_color = good_color if float(cumulative_reward) >= 0.0 else bad_color
    cum_val_text = f"{float(cumulative_reward):+5.2f}"
    _draw_right_aligned_pair(
        ax,
        0.98,
        "Cumulative reward: ",
        cum_val_text,
        cum_val_color,
        base_color,
    )

    # Below the top line, draw a blank spacer,
    # then agent line, then parts, then divider and step total
    dy_axes = _compute_dy_axes(ax)

    def color_for(v: float):
        return good_color if float(v) >= 0.0 else bad_color

    # y cursor: one empty line below top line
    y_ax = 0.98 - dy_axes
    y_ax -= dy_axes  # spacer line

    # Agent line
    # Color only the agent id, keep "Last:" and action number in base color
    segments = [
        (f" #{int(action_num):>2}", base_color),
        (agent_fixed, agent_color),
        ("Last: ", base_color),
    ]
    _draw_right_aligned_segments(ax, 0.98, y_ax, segments)
    y_ax -= dy_axes

    parts_lines = [
        ("deviation: ", float(last_reward.deviation), color_for(last_reward.deviation)),
        ("conflict: ", float(last_reward.conflict), color_for(last_reward.conflict)),
        ("action: ", float(last_reward.action), color_for(last_reward.action)),
    ]

    for label_text, value_num, value_color in parts_lines:
        val_text = f"{value_num:+.3f}"
        _draw_right_aligned_pair(ax, y_ax, label_text, val_text, value_color, base_color)
        y_ax -= dy_axes

    # Bottom line: "= value" for step total
    step_total_val = float(last_reward.total())
    step_total_color = color_for(step_total_val)
    _draw_right_aligned_pair(ax, y_ax, "= ", f"{step_total_val:+.3f}", step_total_color, base_color)


def draw_world_age_overlay(
    ax,
    world_age_seconds: int,
):
    """Top-left overlay showing world age in seconds"""

    base_color = (0.0, 0.0, 0.0, 0.9)
    age_str = f"t = {world_age_seconds} s"

    ax.text(
        0.02,
        0.98,
        age_str,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color=base_color,
        clip_on=True,
    )


# Abbreviations for the on-map action-id markers (actions/actions.py::Action). Speed deltas
# use the configured ±10/±30 kt magnitudes; turn variants encode L/R + rejoin distance (nm)
# with a trailing 's' for the skip-1-waypoint variant; TRB = lengthen trombone; SK = skip
# waypoints; ILS = vector to ILS.
# Marker-id -> abbreviation key for the on-map action markers. Derived from the ACTIVE
# clearance enum, not hard-coded: the ids are set-dependent, so a v1 table drawn beside a v2
# run labels marker "8" as a right turn when it is actually LENGTHEN_TROMBONE. Same reasoning as
# TIER_LABELS below -- anything the legend asserts has to come from the thing it describes.
_ABBREV_BY_NAME = {
    "DO_NOTHING": "NOP",
    # speed (v1 has SMALL/LARGE only; v2 adds MEDIUM and drops SPEED_UP_LARGE)
    "SPEED_UP_SMALL": "SU10", "SPEED_UP_MEDIUM": "SU20", "SPEED_UP_LARGE": "SU30",
    "SLOW_DOWN_SMALL": "SD10", "SLOW_DOWN_MEDIUM": "SD20", "SLOW_DOWN_LARGE": "SD30",
    # turns: v1 parameterises rejoin distance and a skip; v2 keeps only the plain 10 nm rejoin
    "TURN_LEFT_AND_RETURN_TO_ROUTE_5": "L5", "TURN_RIGHT_AND_RETURN_TO_ROUTE_5": "R5",
    "TURN_LEFT_AND_RETURN_TO_ROUTE_10": "L10", "TURN_RIGHT_AND_RETURN_TO_ROUTE_10": "R10",
    "TURN_LEFT_SKIP_1_AND_RETURN_TO_ROUTE_5": "L5s", "TURN_RIGHT_SKIP_1_AND_RETURN_TO_ROUTE_5": "R5s",
    "TURN_LEFT_SKIP_1_AND_RETURN_TO_ROUTE_10": "L10s", "TURN_RIGHT_SKIP_1_AND_RETURN_TO_ROUTE_10": "R10s",
    "TURN_LEFT_AND_RETURN_TO_ROUTE": "TRNL", "TURN_RIGHT_AND_RETURN_TO_ROUTE": "TRNR",
    # trombone: v1 has four absolute levels, v2 one reversible pair
    "LENGTHEN_TROMBONE_1": "TRB1", "LENGTHEN_TROMBONE_2": "TRB2",
    "LENGTHEN_TROMBONE_3": "TRB3", "LENGTHEN_TROMBONE_4": "TRB4",
    "LENGTHEN_TROMBONE": "TRB+", "SHORTEN_TROMBONE": "TRB-",
    "SKIP_1_WAYPOINT": "SK1", "SKIP_2_WAYPOINTS_NOT_NEXT": "SK2",
    "SKIP_3_WAYPOINTS_NOT_NEXT": "SK3", "SKIP_4_WAYPOINTS_NOT_NEXT": "SK4",
    "VECTOR_TO_ILS": "ILS", "DIRECT_TO_MERGE_POINT": "DCT",
}


def _build_action_abbreviations() -> list[tuple[int, str]]:
    from actions.action_set import Action  # local import: action_set reads Config at import time
    return [(int(a), _ABBREV_BY_NAME.get(a.name, a.name[:4].title())) for a in Action]


ACTION_ABBREVIATIONS: list[tuple[int, str]] = _build_action_abbreviations()


def draw_action_legend(ax) -> None:
    """Draw a compact 'number -> abbreviation' key for the on-map action markers.

    Lives on its own narrow axes to the left of the map so the numeric markers scattered on
    the tracks can be decoded to a readable action abbreviation.
    """
    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.5, 0.99, "ACTIONS", ha="center", va="top", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=(0.1, 0.1, 0.1, 1.0))
    ax.text(0.5, 0.968, f"set {Config.ACTION_SET} \u00b7 {len(ACTION_ABBREVIATIONS)}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8.0, family="monospace", color=(0.45, 0.45, 0.45, 1.0))

    n = len(ACTION_ABBREVIATIONS)
    top, bottom = 0.930, 0.008
    for i, (num, abbr) in enumerate(ACTION_ABBREVIATIONS):
        y = top - (top - bottom) * (i / (n - 1))
        ax.text(0.02, y, f"{num:>2}", ha="left", va="center", transform=ax.transAxes,
                fontsize=9.0, family="monospace", color=(0.4, 0.4, 0.4, 1.0))
        ax.text(0.46, y, abbr, ha="left", va="center", transform=ax.transAxes,
                fontsize=9.0, family="monospace", fontweight="bold", color=(0.1, 0.1, 0.1, 1.0))


# Build success-tier labels from Config for the health bar.
TIER_LABELS = ["T1", "T2", "T3", "T4", "T5"]
TIER_CRITERIA = [
    f"{Config.SUCCESS_TIER1_FRAC:.0%}|{Config.SUCCESS_TIER1_DEV_S:.0f}s",
    f"{Config.SUCCESS_TIER2_FRAC:.0%}|{Config.SUCCESS_TIER2_DEV_S:.0f}s",
    f"{Config.SUCCESS_TIER2_FRAC:.0%}|{Config.SUCCESS_TIER3_DEV_S:.0f}s",
    f"{Config.SUCCESS_TIER2_FRAC:.0%}|{Config.SUCCESS_TIER4_DEV_S:.0f}s",
    f"all|{Config.SUCCESS_TIER5_DEV_S:.0f}s",
]


def draw_tier_health_bar(ax, tier: int, metrics: dict | None = None) -> None:
    """Draw a horizontal success-tier "health bar" on its own dedicated axes.

    Five cells (tiers 1..5) fill left-to-right up to the currently achieved ``tier``.
    Achieved cells take a graduated amber->green colour (higher tier = greener), unachieved
    cells are grey. Called every frame, so the bar tracks the agent's predicted success tier
    in real time as it commands aircraft.
    """
    n_tiers = len(TIER_LABELS)
    tier = int(max(0, min(n_tiers, tier)))
    ramp = colormaps["RdYlGn"]
    # Sample amber->green (skip the deep-red low end so T1 doesn't read as a failure).
    achieved_colors = [ramp(v) for v in np.linspace(0.30, 0.90, n_tiers)]
    grey = (0.87, 0.87, 0.87, 1.0)
    grey_edge = (0.62, 0.62, 0.62, 1.0)
    dim_txt = (0.5, 0.5, 0.5, 1.0)

    ax.clear()
    # Reserve the left region [-1.7, 0] for the caption, cells occupy [0, n_tiers].
    ax.set_xlim(-1.7, n_tiers)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    gap = 0.08
    for i in range(n_tiers):
        achieved = (i + 1) <= tier
        is_current = (i + 1) == tier
        color = achieved_colors[i] if achieved else grey
        edge = "black" if is_current else grey_edge
        lw = 2.2 if is_current else 0.8
        ax.add_patch(
            Rectangle(
                (i + gap / 2.0, 0.0),
                1.0 - gap,
                1.0,
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=2,
            )
        )
        txt_color = "black" if achieved else dim_txt
        ax.text(i + 0.5, 0.66, TIER_LABELS[i], ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_color, zorder=3)
        ax.text(i + 0.5, 0.28, TIER_CRITERIA[i], ha="center", va="center",
                fontsize=7.0, family="monospace", color=txt_color, zorder=3)

    # Left caption
    ax.text(-1.6, 0.5, f"SUCCESS\nTIER  {tier}/{n_tiers}", ha="left", va="center",
            fontsize=11, fontweight="bold", color=(0.1, 0.1, 0.1, 1.0), linespacing=1.3)


# --- helpers ---
def within_bounds(x: float, y: float, xy_min: float, xy_max: float) -> bool:
    return (xy_min <= x <= xy_max) and (xy_min <= y <= xy_max)




def draw_label(
    ax,
    x: float,
    y: float,
    text: str,
    color,
    dx: float = 2.0,
    dy: float = 2.0,
    xy_min: float | None = None,
    xy_max: float | None = None,
) -> None:
    """Draw a text label near (x, y), clamped within bounds and clipped to axes."""

    lx = x + dx
    ly = y + dy
    if xy_min is not None and xy_max is not None and not within_bounds(lx, ly, xy_min, xy_max):
        return
    ax.text(lx, ly, text, color=color, fontsize=8, weight="bold", clip_on=True)


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
            self.ax.set_title(title, fontsize=20, fontweight="bold", pad=12)

    # Drawing methods using built-ins with bound injection
    def bounds(self) -> None:
        draw_bounds(self.ax, self.xy_min, self.xy_max)

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
        self, x: float, y: float, color, size: float = 60.0, heading_deg: float | None = None
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

    def actions(
        self, action_history: Sequence[dict], color_map: dict,
        landed_callsigns: set[str] | None = None,
    ) -> None:
        draw_action_markers(
            self.ax,
            action_history,
            color_map,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
            landed_callsigns=landed_callsigns,
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
        agent_id: str,
        action_num: int,
        agent_color,
    ) -> None:
        draw_reward_overlay(
            self.ax,
            cumulative_reward,
            last_parts,
            agent_id=agent_id,
            action_num=action_num,
            agent_color=agent_color,
        )

    def world_age(self, world_age_seconds: int) -> None:
        draw_world_age_overlay(self.ax, world_age_seconds)

    def label(
        self,
        x: float,
        y: float,
        text: str,
        color,
        dx: float = 2.0,
        dy: float = 2.0,
    ) -> None:
        draw_label(
            self.ax,
            x,
            y,
            text,
            color,
            dx=dx,
            dy=dy,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color,
        linewidth: float = 0.5,
        alpha: float = 0.6,
        linestyle: str | tuple = "-",
    ) -> None:
        if len(points) < 2:
            return
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha,
                     linestyle=linestyle, clip_on=True)

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

    def sector_polygons(
        self, polygons: Sequence[Sequence[tuple[float, float]]],
        colors: Sequence,
        alpha: float = 0.08,
    ) -> None:
        draw_sector_polygons(self.ax, polygons, colors, alpha=alpha)


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


def _short_id(callsign: str) -> str:
    """'TEST012' -> '12': the queue number, which is what the eye tracks on a busy plot."""
    digits = "".join(ch for ch in str(callsign) if ch.isdigit())
    return str(int(digits)) if digits else str(callsign)


def _profile_segment(points, t0: float, t1: float):
    """Portion of a (t_s, alt_ft) profile between t0 and t1, with interpolated end points."""
    if len(points) < 2:
        return [], []
    ts = np.array([p[0] for p in points], dtype=float)
    zs = np.array([p[1] for p in points], dtype=float)
    t0, t1 = max(t0, ts[0]), min(t1, ts[-1])
    if t1 <= t0:
        return [], []
    inner = (ts > t0) & (ts < t1)
    seg_t = np.concatenate([[t0], ts[inner], [t1]])
    seg_z = np.interp(seg_t, ts, zs)
    return list(seg_t), list(seg_z)


def draw_time_altitude(ax, data: dict, agent_colors: dict, alt_max_kft: float = 22.0) -> None:
    """Side view of the predicted traffic: altitude against time from now, all flights overlaid.

    ``data`` (built by ATCSingleAgentEnv.side_view_data) holds, all times relative to now:
      * ``profiles``  callsign -> [(t_s, alt_ft), ...] under do-nothing, ending at touchdown
      * ``landed_at`` callsign -> predicted touchdown time (s), for flights landing in view
      * ``targets``   callsign -> AMAN target landing time (s)
      * ``events``    [(t_s, callsign_a, callsign_b, severity)] predicted separation samples
      * ``highlight`` callsigns drawn at full strength (the window), or None for all
      * ``horizon_s`` the prediction horizon

    Infringements are drawn ON the profiles of both aircraft involved, over the minute each
    sample covers: solid red where severity reaches 1 (a loss of separation: <= 3 NM and
    <= 500 ft), dashed orange where 0 < severity < 1 (the 3-5 NM band). A vertical connector
    would not work -- a loss of separation means the two altitudes are within 500 ft, so it
    would be invisible exactly when it matters. Altitude is floored at 0.
    """
    from matplotlib.lines import Line2D
    from utils.infringement_utils import is_realised_los

    ax.clear()
    profiles = data.get("profiles", {})
    highlight = data.get("highlight")
    horizon_min = max(1.0, float(data.get("horizon_s", 3600.0)) / 60.0)
    grey = (0.4, 0.4, 0.4, 1.0)

    for cs, pts in profiles.items():
        if not pts:
            continue
        color = agent_colors.get(cs, grey)
        faded = highlight is not None and cs not in highlight
        t = [p[0] / 60.0 for p in pts]
        z = [max(0.0, p[1]) / 1000.0 for p in pts]
        ax.plot(t, z, color=color, linewidth=0.8 if faded else 1.4, alpha=0.3 if faded else 0.95,
                zorder=2)
        if t[0] <= 0.01:  # airborne now: mark its current altitude
            ax.scatter([t[0]], [z[0]], color=color, s=14, alpha=0.3 if faded else 1.0, zorder=3)
        if not faded:
            ax.text(t[0], z[0] + 0.35, _short_id(cs), color=color, fontsize=6.5,
                    ha="center", va="bottom", zorder=5)
        t_land = data.get("landed_at", {}).get(cs)
        if t_land is not None:
            ax.scatter([t_land / 60.0], [0.0], marker="v", color=color, s=26,
                       alpha=0.35 if faded else 1.0, zorder=4, clip_on=False)

    # AMAN target landing times: a coloured tick on the ground line. The horizontal gap to the
    # same flight's touchdown marker is its predicted deviation.
    for cs, t_tgt in data.get("targets", {}).items():
        if 0.0 <= t_tgt <= horizon_min * 60.0 and cs in profiles:
            faded = highlight is not None and cs not in highlight
            ax.plot([t_tgt / 60.0] * 2, [0.0, 0.9], color=agent_colors.get(cs, grey),
                    linewidth=2.2, alpha=0.35 if faded else 1.0, zorder=3, solid_capstyle="butt")

    # Predicted separation samples, drawn on both profiles. Collected into two LineCollections
    # (one draw call each) rather than one plot() per segment: a busy frame has 100+ segments.
    from matplotlib.collections import LineCollection

    half = 30.0  # records arrive ~every 60 s; each covers the minute around it
    worst = {}
    los_segs, band_segs, band_colors = [], [], []
    for t_s, a, b, sev in data.get("events", []):
        if sev <= 0.0 or not (0.0 <= t_s <= horizon_min * 60.0):
            continue
        los = is_realised_los(sev)
        for cs in (a, b):
            seg_t, seg_z = _profile_segment(profiles.get(cs, []), t_s - half, t_s + half)
            if len(seg_t) >= 2:
                seg = [(x / 60.0, max(0.0, v) / 1000.0) for x, v in zip(seg_t, seg_z)]
                if los:
                    los_segs.append(seg)
                else:
                    band_segs.append(seg)
                    band_colors.append((1.0, 0.55, 0.0, 0.4 + 0.5 * sev))
        key = tuple(sorted((a, b)))
        if key not in worst or sev > worst[key][1]:
            worst[key] = (t_s, sev)
    if band_segs:
        ax.add_collection(LineCollection(band_segs, colors=band_colors, linewidths=2.2,
                                         linestyles=[(0, (3, 2))], zorder=5, capstyle="butt"))
    if los_segs:
        ax.add_collection(LineCollection(los_segs, colors=[(1.0, 0.0, 0.0, 0.9)],
                                         linewidths=3.2, zorder=6, capstyle="butt"))

    # One label per loss-of-separation pair at its worst sample, nudged up when it would land
    # on an earlier label (pairs sharing an aircraft often peak at nearly the same point).
    placed = []
    for (a, b), (t_s, sev) in sorted(worst.items(), key=lambda kv: kv[1][0]):
        pa = profiles.get(a, [])
        if not is_realised_los(sev) or len(pa) < 2:
            continue
        z = float(np.interp(t_s, [p[0] for p in pa], [p[1] for p in pa])) / 1000.0
        x, y = t_s / 60.0, max(0.0, z) + 0.8
        while any(abs(x - px) < 4.0 and abs(y - py) < 0.9 for px, py in placed):
            y += 0.9
        placed.append((x, y))
        ax.text(x, y, f"LoS {_short_id(a)}/{_short_id(b)}", color="red", fontsize=6.5,
                fontweight="bold", ha="center", zorder=7)

    ax.set_xlim(0.0, horizon_min)
    ax.set_ylim(0.0, alt_max_kft)
    ax.set_xlabel("time from now (min)", fontsize=7)
    ax.set_ylabel("altitude (kft)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_title("Predicted profiles under do-nothing (side view)", fontsize=8.5)
    handles = [
        Line2D([], [], color="red", linewidth=3.2, label="loss of separation (severity 1)"),
        Line2D([], [], color="darkorange", linewidth=2.2, linestyle=(0, (3, 2)),
               label="3-5 NM (0 < severity < 1)"),
        Line2D([], [], color=grey, marker="v", linestyle="none", label="predicted touchdown"),
        Line2D([], [], color=grey, linewidth=2.2, label="AMAN target (on ground line)"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.85)

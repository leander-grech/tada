"""
This module provides pure functions to assemble RGB renders for ATC environments,
allowing an environment to call into it with the necessary state.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from actions.rewards import get_aircraft_time_deviations
from config.config import Config
from flight_simulator import World
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from simulator.simulator import Simulator
from utils.infringement_utils import InfringementTuple, get_world_infringements
import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
from utils.plot_utils import (
    BoundedPlot,
    draw_action_legend,
    draw_airport_3d,
    draw_info_panel,
    draw_tier_health_bar,
    draw_time_altitude,
)  # , sector_polygons
from utils.reward_utils import RewardParts
from atc_env.sector_selection import resolve_render_sector_polygons


def init_plot(title: Optional[str] = None, approach_view: str = "time"):
    """Create the figure layout and return (fig, map-plot, legend_ax, health_ax, ax3d, info_ax).

    Layout (top row): [action-legend | main 2D map | 3D approach view over aircraft info panel];
    bottom row: success-tier health-bar strip spanning the full width.

    ``approach_view`` picks the top-right panel: "time" (2D altitude-vs-time side view; the
    returned ``ax3d`` is then an ordinary axes) or "3d" (the 3D approach view)."""

    fig = plt.figure(figsize=(13.5, 9.0), dpi=150)
    if fig.canvas is None or not hasattr(fig.canvas, "buffer_rgba"):
        fig.set_canvas(FigureCanvas(fig))
    # The right column carries the widest content (the info panel's fixed-width columns), so
    # it gets the extra width the map's empty corners were spending.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 8.2, 6.5], height_ratios=[16.0, 1.1])
    legend_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    # Right column: 3D approach view (top) over the aircraft info panel (bottom).
    right_gs = gs[0, 2].subgridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.06)
    ax3d = fig.add_subplot(right_gs[0], projection="3d" if approach_view == "3d" else None)
    info_ax = fig.add_subplot(right_gs[1])
    health_ax = fig.add_subplot(gs[1, :])
    plot = BoundedPlot(ax, Config.RENDER_POS_XY_MIN, Config.RENDER_POS_XY_MAX)
    plot.set_limits_and_grid(title=title)
    # Zoom the map view in slightly and offset it upward (the airport/merge area sits above
    # centre and the outer sector is mostly empty). Drawing still clips to the full bounds.
    ax.set_xlim(-86.0, 86.0)
    ax.set_ylim(-78.0, 94.0)
    return fig, plot, legend_ax, health_ax, ax3d, info_ax


def finalize_rgb(fig: Figure) -> np.ndarray:
    """Render the matplotlib figure and return an RGB numpy array.

    The caller is responsible for any layout (tight_layout / axes repositioning) before
    calling this — it only rasterises."""

    canvas = cast(FigureCanvas, fig.canvas)
    canvas.draw()
    renderer = canvas.get_renderer()
    width, height = int(renderer.width), int(renderer.height)
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape((height, width, 4))
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def make_interstitial_frames(
    width: int,
    height: int,
    title: str,
    lines: Sequence[str],
    n_frames: int = 3,
    dpi: int = 150,
) -> List[np.ndarray]:
    """Build ``n_frames`` identical black cards (white text) sized to (height, width).

    Used to separate episodes in a stitched render with a short black title card showing the
    episode's chosen parameters (seed, deterministic/stochastic, shield, ...).
    """
    fig = Figure(figsize=(width / float(dpi), height / float(dpi)), dpi=dpi)
    fig.set_canvas(FigureCanvas(fig))
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("black")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.5, 0.62, title, color="white", ha="center", va="center",
            fontsize=28, fontweight="bold")
    if lines:
        # Left-align the block (at a roughly centred x) so padded label colons line up.
        ax.text(0.34, 0.43, "\n".join(lines), color="white", ha="left", va="center",
                fontsize=15, family="monospace", linespacing=1.8)

    canvas = cast(FigureCanvas, fig.canvas)
    canvas.draw()
    renderer = canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return [rgb.copy() for _ in range(max(1, int(n_frames)))]


def compute_infringement_summary(
    simulator: Simulator,
    possible_agents: Sequence[str],
    predicted_world: Optional[World] = None,
) -> Dict[str, InfringementTuple]:
    """Per-aircraft *closest* conflict partner still ahead of the aircraft.

    Read from ``predicted_world`` -- the committed DO_NOTHING rollout -- rather than from what
    has already happened, because that is the conflict the frame's reward overlay is charging
    for and the only one the agent can still do something about. A realised loss of separation
    ends the episode on the spot, so a panel that only reported live pairs would spend almost
    every frame reporting nothing while the conflict term ran a deficit. Only records
    timestamped at or after the current world age are kept, so the rollout's inherited history
    does not leak in. Falls back to the live world when no forecast is supplied.

    Ranked by ``proximity`` (the wide 3-10 NM detection band), NOT by ``value``: a pair closing
    inside 10 NM is what the info panel is there to warn about, and ranking by the charged
    severity would silently drop every such pair -- severity is 0 beyond 5 NM, so the panel
    could only ever show a conflict that was already too late to prevent.

    The returned tuple still carries the charged ``value`` (severity), so the panel reports
    both: how close the pair is predicted to get, and what that closeness actually costs.
    """

    assert simulator.world is not None
    callsign_set = set(possible_agents)
    completed_callsigns = {ac.call_sign for ac in simulator.world.aircraft if ac.mission.completed}

    now = float(simulator.world.age_in_seconds)
    if predicted_world is not None:
        source = predicted_world
        window_start = now
        horizon = max(0.0, float(predicted_world.age_in_seconds) - now)
    else:
        source = simulator.world
        window_start = now - float(Config.TIME_BETWEEN_ACTIONS)
        horizon = float(Config.TIME_BETWEEN_ACTIONS)

    infr_map = get_world_infringements(
        source, current_time_s=window_start, horizon_time_s=horizon
    )

    closest: Dict[str, InfringementTuple] = {}
    for cs, tuples in infr_map.items():
        if cs in completed_callsigns or cs not in callsign_set or not tuples:
            continue
        closest[cs] = max(tuples, key=lambda t: (float(t.proximity), float(t.value)))

    return closest


# One shared column layout for the aircraft info panel: the header and every row are formatted
# through it, so the columns line up by construction rather than by hand-counted padding.
_INFO_ROW_FMT = (
    "{cs:<7} {st:<3} {spd:>4} {x:>6} {y:>6} {alt:>6} {dev:>5} {act:>3} "
    "{sev:>4} {sep:>4} {eta:>5} {pair:<7}"
)
INFO_HEADER = _INFO_ROW_FMT.format(
    cs="AC", st="ST", spd="KT", x="X", y="Y", alt="ALT", dev="DEV", act="ACT",
    sev="SEV", sep="SEP", eta="IN", pair="PAIR",
).rstrip()
# Spelled out under the table: SEV/SEP/IN/PAIR are a forecast, not a measurement, and reading
# them as "how close this aircraft is right now" would be wrong.
INFO_FOOTNOTE = "SEV/SEP/IN/PAIR: worst predicted conflict ahead (severity, nm, s, with)"


def _info_row(
    callsign: str,
    status: str,
    tdev_s: float,
    n_act: int,
    pos: Optional[Tuple[float, float, float]],
    speed_knots: Optional[float],
    conflict: Optional[InfringementTuple],
    now_s: float = 0.0,
) -> str:
    """Format one aircraft's info-panel row.

    ``pos``/``speed_knots`` are None when the aircraft has no live state to report, in which
    case the state columns show dashes but the deviation and clearance count -- which stay
    meaningful -- do not. ``status`` is what tells those two silences apart: PRE (not yet
    under control) vs LND (landed)."""
    if pos is None:
        spd = x = y = alt = "-"
    else:
        spd = f"{float(speed_knots or 0.0):.0f}"
        x, y, alt = f"{pos[0]:.1f}", f"{pos[1]:.1f}", f"{pos[2]:.0f}"

    if conflict is None:
        sev = sep = eta = "-"
        pair = "-"
    else:
        sev = f"{float(conflict.value):.2f}"
        sep = f"{float(conflict.horizontal_nm):.1f}"
        eta = f"{max(0.0, float(conflict.timestamp) - float(now_s)):.0f}"
        pair = str(conflict.counterpart)[:7]

    return _INFO_ROW_FMT.format(
        cs=callsign[:7], st=status, spd=spd, x=x, y=y, alt=alt,
        dev=f"{tdev_s:+.0f}", act=f"{int(n_act)}", sev=sev, sep=sep, eta=eta, pair=pair,
    ).rstrip()


def render_past_tracks_and_passed_wps(
    simulator: Simulator,
    past_tracks: Dict[str, List[Tuple[float, float]]],
    initial_routes: Dict[str, List[str]],
    agent_colors: Dict[str, Tuple[float, float, float, float]],
    plot: BoundedPlot,
) -> None:
    """Render past aircraft tracks and already-passed flight plan waypoints."""

    assert simulator.world is not None

    # Past tracks
    for callsign, past_points in past_tracks.items():
        if past_points and len(past_points) >= 2:
            color = agent_colors.get(callsign)
            plot.polyline(past_points, color=color, linewidth=0.5, alpha=0.6)

    # Passed flight-plan waypoints (no connecting line)
    for callsign, init_names in initial_routes.items():
        color = agent_colors.get(callsign)
        ac = next((a for a in simulator.world.aircraft if a.call_sign == callsign), None)
        if ac is None:
            continue
        remaining_names = {rp.waypoint_name for rp in ac.route} if ac.route else set()
        passed_names = [n for n in init_names if n not in remaining_names]

        past_wp_points = [
            (float(wp.position_nm.x), float(wp.position_nm.y))
            for n in passed_names
            if (wp := simulator.world.waypoints.get(n)) is not None
        ]

        if past_wp_points:
            plot.waypoints(
                past_wp_points,
                color=color,
                waypoint_size=6.0,
                waypoint_alpha=0.4,
            )


def render_aircraft_and_collect_info(
    simulator: Simulator,
    prev_world: World,
    possible_agents: Sequence[str],
    agent_colors: Dict[str, Tuple[float, float, float, float]],
    plot: BoundedPlot,
    action_counts: Optional[Dict[str, int]] = None,
    predicted_tracks: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
) -> Tuple[List[str], List, str]:
    """Render aircraft states, routes, and collect info-panel rows.

    ``action_counts`` maps callsign -> number of (non DO_NOTHING) clearances the agent has
    issued to that aircraft so far this episode; shown in the info panel as the ``ACT`` column.
    ``predicted_tracks`` maps callsign -> the aircraft's forecast [(x, y, alt_ft), ...] under
    DO_NOTHING, drawn as a dotted look-ahead from its current position.
    Returns (info_rows, row_colors, summary) where summary is an averages line (mean action
    count and mean absolute time deviation over all aircraft, 2 dp)."""

    assert simulator.world is not None
    assert prev_world is not None

    info_rows: List[str] = []
    row_colors: List = []

    action_counts = action_counts or {}
    predicted_tracks = predicted_tracks or {}
    time_dev_map = get_aircraft_time_deviations(prev_world)

    # Worst conflict still ahead of each aircraft, read from the same forecast the reward's
    # conflict term is charged against (`prev_world` is the committed DO_NOTHING rollout).
    now_s = float(simulator.world.age_in_seconds)
    conflicts = compute_infringement_summary(simulator, possible_agents, prev_world)

    for callsign in sorted(possible_agents):
        ac = simulator.world.get_aircraft_by_callsign(callsign)
        tdev_s = float(time_dev_map.get(callsign, 0.0))
        n_act = int(action_counts.get(callsign, 0))

        if ac is not None and not ac.mission.completed and ac.under_control:
            pos = ac.state.position_nm_nm_ft
            vel = ac.state.velocity_knots
            x, y = float(pos.x), float(pos.y)

            color = agent_colors.get(callsign, tuple(np.random.rand(3)))

            # Build route points: start from current position, then planned waypoints
            route_points: List[Tuple[float, float]] = [(x, y)]
            if ac.route:
                for route_point in ac.route:
                    wp = simulator.world.waypoints.get(route_point.waypoint_name)
                    if wp is not None:
                        route_points.append((float(wp.position_nm.x), float(wp.position_nm.y)))

            # Draw route polyline and waypoints
            plot.route(route_points, color=color, include_line=True, include_waypoints=True)

            # Predicted trajectory (dotted): where this aircraft ends up if nothing else is
            # commanded. Distinct from the route above, which is only the plan it is flying.
            forecast = predicted_tracks.get(callsign) or []
            if len(forecast) >= 2:
                plot.polyline(
                    [(p[0], p[1]) for p in forecast],
                    color=color, linewidth=1.0, alpha=0.55, linestyle=":",
                )

            # Draw current aircraft position and velocity
            heading_deg = float(np.degrees(np.arctan2(float(vel.y), float(vel.x))))
            plot.triangle(x, y, color=color, size=40.0, heading_deg=heading_deg)
            plot.velocity(x, y, float(vel.x), float(vel.y), color=color, target_len=10.0)

            info_rows.append(_info_row(
                callsign, "AIR", tdev_s, n_act,
                pos=(x, y, float(pos.z)), speed_knots=float(vel.length()),
                conflict=conflicts.get(callsign), now_s=now_s,
            ))
            row_colors.append(color)
        else:
            # Landed, or not yet under control: the state columns have nothing to say, but
            # the final deviation and the clearance tally still do.
            status = "LND" if (ac is not None and ac.mission.completed) else "PRE"
            info_rows.append(_info_row(
                callsign, status, tdev_s, n_act, pos=None, speed_knots=None, conflict=None,
            ))
            row_colors.append(agent_colors.get(callsign))

    # Averages over all aircraft (2 dp): mean issued-clearance count and mean |deviation|.
    callsigns = list(sorted(possible_agents))
    n_ac = len(callsigns)
    if n_ac:
        avg_act = sum(int(action_counts.get(cs, 0)) for cs in callsigns) / n_ac
        avg_dev = sum(abs(float(time_dev_map.get(cs, 0.0))) for cs in callsigns) / n_ac
    else:
        avg_act = avg_dev = 0.0
    summary = f"AVG:  actions {avg_act:.2f}   |dev| {avg_dev:.2f} s"

    return info_rows, row_colors, summary


def ensure_agent_colors(
    agent_ids: Iterable[str],
    existing: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return a mapping for all agent_ids with stable colors (tab10)."""

    cmap = plt.get_cmap("tab10")
    result = dict(existing or {})
    for i, aid in enumerate(agent_ids):
        if aid not in result:
            result[aid] = to_rgba(cmap(i % 10))
    return result


def render_frame(
    *,
    simulator: Simulator,
    prev_world: World,
    possible_agents: Sequence[str],
    agent_colors: Dict[str, Tuple[float, float, float, float]],
    past_tracks: Dict[str, List[Tuple[float, float]]],
    initial_routes: Dict[str, List[str]],
    action_history: Sequence[Dict[str, Any]],
    cumulative_reward: float,
    last_reward: RewardParts,
    last_actor_id: str,
    last_action_num: Optional[int],
    title: Optional[str] = None,
    tier: int = 0,
    tier_metrics: Optional[Dict[str, Any]] = None,
    action_counts: Optional[Dict[str, int]] = None,
    approach_xy: Optional[Tuple[float, float]] = None,
    predicted_tracks: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    approach_zoom_nm: float = 35.0,
    approach_view: str = "time",
    side_view: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Assemble a full RGB frame with overlays identical to multi-agent env.

    ``tier`` is the currently-achieved success tier drawn as a health bar below the map;
    ``action_counts`` maps callsign -> issued-clearance count shown in the info panel;
    ``approach_xy`` centres the 3D airport-vicinity view (half-width ``approach_zoom_nm``);
    ``predicted_tracks`` maps callsign -> forecast [(x, y, alt_ft), ...] drawn on both views.
    ``approach_view`` "time" replaces the 3D view with a side view built from ``side_view``
    (see plot_utils.draw_time_altitude); "3d" keeps the 3D approach view."""

    if approach_view == "time" and side_view is None:
        approach_view = "3d"  # nothing to draw the side view from
    fig, plot, legend_ax, health_ax, ax3d, info_ax = init_plot(title, approach_view)

    # Action-id abbreviation key (left column)
    draw_action_legend(legend_ax)

    # Past tracks and passed waypoints
    render_past_tracks_and_passed_wps(simulator, past_tracks, initial_routes, agent_colors, plot)

    polygons, polygon_colors = resolve_render_sector_polygons()

    plot.sector_polygons(polygons, polygon_colors, alpha=0.2)

    # Aircraft and info rows (info rendered on the side panel, not over the map)
    info_rows, row_colors, info_summary = render_aircraft_and_collect_info(
        simulator, prev_world, possible_agents, agent_colors, plot,
        action_counts=action_counts, predicted_tracks=predicted_tracks,
    )

    # Action markers and frame bounds. Landed aircraft collapse to dots (no number) so their
    # history stops obstructing the numbers of aircraft still being worked.
    landed_callsigns = {ac.call_sign for ac in simulator.world.aircraft if ac.mission.completed}
    plot.actions(action_history, agent_colors, landed_callsigns=landed_callsigns)
    plot.bounds()

    # World age overlay (top-left of map)
    assert simulator.world is not None
    plot.world_age(int(simulator.world.age_in_seconds))

    # Reward overlay in top-right corner
    plot.reward(
        cumulative_reward,
        last_reward,
        last_actor_id or "-------",
        int(last_action_num) if last_action_num is not None else -1,
        agent_color=agent_colors.get(last_actor_id),
    )

    # 3D approach view (right column, top) + aircraft info panel (right column, bottom)
    if approach_view == "3d":
        draw_airport_3d(
            ax3d, simulator.world, agent_colors,
            approach_xy if approach_xy is not None else (0.0, 0.0),
            past_tracks=past_tracks,
            predicted_tracks=predicted_tracks,
            zoom_nm=float(approach_zoom_nm),
        )
    else:
        draw_time_altitude(ax3d, side_view, agent_colors)
    draw_info_panel(
        info_ax, info_rows, row_colors=row_colors, summary=info_summary,
        header=INFO_HEADER, footnote=INFO_FOOTNOTE,
    )

    # Success-tier health bar strip below everything
    draw_tier_health_bar(health_ax, int(tier), tier_metrics)

    # Layout, then nudge the 3D cube right within its column (mplot3d leaves it
    # left-justified) -- but only a little: mplot3d draws the z-axis tick labels OUTSIDE the
    # axes box on its right, so an offset large enough to centre the cube pushes those labels
    # off the figure. h_pad keeps the health-bar strip clear of the map's x-axis label/ticks.
    fig.tight_layout(h_pad=2.6, w_pad=1.0)
    if approach_view == "3d":
        _p3d = ax3d.get_position()
        ax3d.set_position([_p3d.x0 + 0.010, _p3d.y0, _p3d.width * 0.88, _p3d.height])
    else:
        # The map's title is one long line that runs to the right of the map; keep the side
        # view's top level with the map axes so it does not cover the end of that title.
        _ps, _top = ax3d.get_position(), plot.ax.get_position().y1
        if _ps.y1 > _top:
            ax3d.set_position([_ps.x0, _ps.y0, _ps.width, _top - _ps.y0])

    # Finalize and return RGB array
    return finalize_rgb(fig)

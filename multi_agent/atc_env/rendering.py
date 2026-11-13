"""EnvRenderingMixin for ``ATCEnv``; matplotlib loads only when a frame is drawn."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from actions.actions import (
    count_nonzero_actions,
    filter_render_action_history,
)
from actions.reward_parts import build_acted_aircraft_reward_rows
from atc_env.coverage_consumers import (
    format_timing,
    proxy_map,
    timing_map,
    timing_report,
)
from atc_env.environment_helpers import (
    format_infr_field,
    get_action_selection_mode,
    resolve_route_point_position_nm,
    resolve_waypoint_name_position_nm,
)
from outcomes.infringement import (
    MostProminent,
    filter_infringements_by_window,
    get_world_infringements,
)

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv
    from atc_env.render_plots import BoundedPlot


def agent_colors(callsigns) -> dict[str, tuple[float, float, float, float]]:
    """Stable per-callsign colours from the tab10 cycle, in agent order."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    cmap = plt.get_cmap("tab10")
    return {callsign: to_rgba(cmap(i % 10)) for i, callsign in enumerate(callsigns)}


class EnvRenderingMixin:
    def _init_plot(self: "ATCEnv"):
        """Create a fresh figure and bounded plot for rendering."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        from atc_env.render_plots import BoundedPlot

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        if fig.canvas is None or not hasattr(fig.canvas, "buffer_rgba"):
            fig.set_canvas(FigureCanvas(fig))
        plot = BoundedPlot(
            ax, self.settings.env.render_pos_xy_min, self.settings.env.render_pos_xy_max
        )
        title = f"TADA evaluation (seed = {self.scenario_seed})"
        plot.set_limits_and_grid(title=title)
        return fig, plot

    def _resolve_render_sector_polygons(
        self: "ATCEnv",
    ) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float, float, float]]]:
        """Resolve selected-sector polygons once per episode for rendering."""
        if get_action_selection_mode(self.settings.core.selection) != "sector_based":
            return [], []

        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        selected_index = self._resolve_selected_group_index()
        cmap = plt.get_cmap("Set2")
        group_colors = {
            group: to_rgba(cmap(idx % cmap.N))
            for idx, group in enumerate(selected_index.selected_groups)
        }

        # Render in configured group order so later groups are drawn on top and
        # visually claim overlapping areas consistently with selection precedence.
        polygons: list[list[tuple[float, float]]] = []
        colors: list[tuple[float, float, float, float]] = []
        for group in selected_index.selected_groups:
            group_color = group_colors[group]
            for sector_name in group:
                polygon = [
                    (float(point[0]), float(point[1]))
                    for point in selected_index.sector_defs[sector_name]["polygon"]
                ]
                polygons.append(polygon)
                colors.append(group_color)

        return polygons, colors

    def _compute_infringement_summary(self: "ATCEnv") -> dict[str, MostProminent]:
        """
        Return per-aircraft max_value for loss of
        separation infringements, limited to given callsigns.
        """

        assert self.simulator.world is not None, "Simulator world is not initialized"
        # Use episode's aircraft (from _initial_routes), not all possible agents
        callsign_set = set(self._initial_routes.keys())
        completed_callsigns = {
            ac.call_sign for ac in self.simulator.world.aircraft if ac.mission.completed
        }

        most_prominent: dict[str, MostProminent] = {cs: MostProminent(0.0) for cs in callsign_set}

        window_start = float(self.simulator.world.age_in_seconds) - float(
            self.settings.env.time_between_actions
        )
        window_end = window_start + float(self.settings.env.time_between_actions)
        infr_map = filter_infringements_by_window(
            get_world_infringements(self.simulator.world, self.settings.core.safety),
            window_start,
            window_end,
        )

        for cs, tuples in infr_map.items():
            if cs in completed_callsigns or cs not in callsign_set or not tuples:
                continue

            infr_tuple = max(tuples, key=lambda x: float(x.value))
            if float(infr_tuple.value) > most_prominent[cs].value:
                most_prominent[cs].value = float(infr_tuple.value)
                if infr_tuple.counterpart and infr_tuple.counterpart != cs:
                    most_prominent[cs].counterpart.update([str(infr_tuple.counterpart)])

        return most_prominent

    def _render_past_tracks_and_passed_wps(self: "ATCEnv", plot: BoundedPlot):
        """Render past aircraft tracks and already-passed flight plan waypoints."""

        assert self.simulator.world is not None, "Simulator world is not initialized"

        # Past tracks
        for callsign, past_points in self._past_tracks.items():
            if past_points and len(past_points) >= 2:
                color = self._agent_colors.get(callsign)
                plot.polyline(past_points, color=color, linewidth=0.5, alpha=0.6)

        # Passed flight-plan waypoints (no connecting line)
        for callsign, init_names in self._initial_routes.items():
            color = self._agent_colors.get(callsign)
            ac = next(
                (a for a in self.simulator.world.aircraft if a.call_sign == callsign),
                None,
            )
            if ac is None:
                continue
            remaining_names = {rp.waypoint_name for rp in ac.route} if ac.route else set()
            passed_names = [n for n in init_names if n not in remaining_names]

            past_wp_points: list[tuple[float, float]] = []
            for waypoint_name in passed_names:
                waypoint_xy = resolve_waypoint_name_position_nm(
                    waypoint_name, self.simulator.world.waypoints
                )
                if waypoint_xy is not None:
                    past_wp_points.append(waypoint_xy)

            if past_wp_points:
                plot.waypoints(
                    past_wp_points,
                    color=color,
                    waypoint_size=6.0,
                    waypoint_alpha=0.4,
                )

    def _render_aircraft_and_collect_info(
        self: "ATCEnv", plot: BoundedPlot, deviation_proxies: dict[str, float]
    ):
        """Render aircraft states, routes, and collect info-box rows."""

        assert self.simulator.world is not None, "Simulator world is not initialized"
        assert self.prev_world is not None, "Previous world is not initialized"

        info_rows: list[str] = []
        row_colors: list = []

        # Precompute infringement best values and participant IDs per aircraft
        infr_most_prominent = self._compute_infringement_summary()

        # Use episode's aircraft (from _initial_routes), not all possible agents
        aircraft_by_callsign = {ac.call_sign: ac for ac in self.simulator.world.aircraft}
        for callsign in (target.callsign for target in self.arrival_targets):
            ac = aircraft_by_callsign.get(callsign)
            if ac is not None and not ac.mission.completed and ac.under_control:
                pos = ac.state.position_nm_nm_ft
                vel = ac.state.velocity_knots
                x, y = float(pos.x), float(pos.y)

                color = self._agent_colors.get(callsign, tuple(np.random.rand(3)))

                # Build route points: start from current position, then planned waypoints
                route_points: list[tuple[float, float]] = [(x, y)]
                if ac.route:
                    for route_point in ac.route:
                        route_xy = resolve_route_point_position_nm(
                            route_point, self.simulator.world.waypoints
                        )
                        if route_xy is not None:
                            route_points.append(route_xy)

                # Draw route polyline and waypoints
                plot.route(route_points, color=color, include_line=True, include_waypoints=True)

                # Draw current aircraft position and velocity
                heading_deg = float(np.degrees(np.arctan2(float(vel.y), float(vel.x))))
                plot.triangle(x, y, color=color, size=40.0, heading_deg=heading_deg)
                plot.velocity(x, y, float(vel.x), float(vel.y), color=color, target_len=10.0)

                # Collect info box row
                speed_knots = vel.length()
                timing_text = format_timing(timing_map(self.forecast_coverage).get(callsign))
                mp = infr_most_prominent.get(callsign, MostProminent(0.0))
                infr_field = format_infr_field(mp.value, ",".join(sorted(mp.counterpart)))
                info_line = (
                    f"{callsign}: {speed_knots:3.0f} kt | "
                    f"({x:5.1f},{y:5.1f},{float(pos.z):5.0f}) | "
                    f"{timing_text} | {infr_field}"
                )
                info_rows.append(info_line)
                row_colors.append(color)
            else:
                # Completed or missing: keep row, show placeholders with final deviation
                timing_text = format_timing(timing_map(self.forecast_coverage).get(callsign))
                # Show dashed speed/alt and an empty fixed-width infringement field
                empty_infr_field = format_infr_field(0.0, "")
                info_line = (
                    f"{callsign}: {'---':>3} kt | ({'-----':>5},{'-----':>5},{'-----':>5}) | "
                    f"{timing_text} | {empty_infr_field}"
                )
                info_rows.append(info_line)
                row_colors.append(self._agent_colors.get(callsign))

        return info_rows, row_colors

    def _finalize_rgb(self: "ATCEnv", fig):
        """Render the matplotlib figure and return an RGB numpy array."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig.tight_layout()
        canvas = cast(FigureCanvas, fig.canvas)
        canvas.draw()
        # Use renderer pixel dimensions (accounts for HiDPI/Retina scaling)
        renderer = canvas.get_renderer()
        width, height = int(renderer.width), int(renderer.height)
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape((height, width, 4))
        rgb = rgba[:, :, :3].copy()
        plt.close(fig)
        return rgb

    def render(self: "ATCEnv", mode="rgb_array"):
        """Render current ATC situation and return an RGB numpy array."""

        assert mode == "rgb_array", "Only rgb_array mode is supported for recording"
        assert self.simulator.world is not None, "Simulator world is not initialized"
        assert self.prev_world is not None, "Previous world is not initialized"

        self._agent_colors = agent_colors(self._agent_color_order)
        # Create figure and plot
        fig, plot = self._init_plot()

        deviation_proxies = proxy_map(self.forecast_coverage)
        report = timing_report(self.forecast_coverage)

        if self._sector_polygons:
            plot.sector_polygons(self._sector_polygons, self._sector_polygon_colors, alpha=0.2)

        # Draw past tracks and passed waypoints
        self._render_past_tracks_and_passed_wps(plot)

        # Draw aircraft and collect info box contents
        info_rows, row_colors = self._render_aircraft_and_collect_info(plot, deviation_proxies)

        # Draw action markers (persist across frames), excluding "do nothing" (action 0)
        active_actions = filter_render_action_history(self._action_history)
        plot.actions(active_actions, self._agent_colors)

        # Draw a fixed boundary box for clarity
        plot.bounds()

        # Draw speed/altitude info box in lower-right corner (preserve row order)
        plot.info(info_rows, row_colors=row_colors)

        tdt_line = (
            f"timing: exact {report['exact_count']}, interval {report['interval_count']}, "
            f"unknown {report['unknown_count']}, sign-ambiguous {report['sign_ambiguous_count']}"
        )
        exact_mean = report["exact_mean_absolute_error_s"]
        td_line = (
            "exact arrival mean: unknown"
            if exact_mean is None
            else f"exact arrival mean |error|: {exact_mean:.1f} s"
        )
        nz_total, nz_ex_trb, _speed = count_nonzero_actions(self._action_history)
        nz_line = f"nza = {nz_ex_trb} ({nz_total})"

        # Draw world age and time-deviation overlay (top-left)
        plot.world_age(
            int(self.simulator.world.age_in_seconds),
            extra_lines=[tdt_line, td_line, nz_line],
        )

        # Reward overlay in top-right corner
        acted_rows = build_acted_aircraft_reward_rows(self.last_step_action_nums)

        plot.reward(
            self.cumulative_reward,
            self.last_reward,
            acted_rows=acted_rows,
            color_map=self._agent_colors,
        )

        # Finalize and return RGB array
        return self._finalize_rgb(fig)

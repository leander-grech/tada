"""Multi-page PDF rendering of the offline evaluation report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from evaluation.report.build import (
    build_summary,
    normalize_episode_df,
    normalize_step_df,
    safe_float,
    safe_int,
    select_representative_episodes,
)


def _flatten_action_values(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [safe_int(item) for item in value]
    if isinstance(value, str) and value:
        pieces = [piece for piece in value.split(",") if piece]
        return [safe_int(piece) for piece in pieces]
    return []


def _render_message_page(pdf: PdfPages, *, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left")
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=ax.transAxes,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_overview_page(
    pdf: PdfPages,
    *,
    episode_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Evaluation Report Overview", fontsize=18, x=0.05, ha="left")

    axes[0].axis("off")
    metadata_lines = [
        f"checkpoint: {metadata.get('checkpoint', '?')}",
        f"episodes: {metadata.get('episodes', len(episode_df))}",
        f"max_aircraft: {metadata.get('max_aircraft', '?')}",
        f"max_steps: {metadata.get('max_steps', '?')}",
        f"generated_at: {metadata.get('generated_at', '?')}",
    ]
    axes[0].text(
        0.02,
        0.98,
        "\n".join(metadata_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=axes[0].transAxes,
    )

    outcome_counts = summary.get("outcomes", {})
    if outcome_counts:
        labels = list(outcome_counts.keys())
        values = [int(outcome_counts[label]) for label in labels]
        axes[1].bar(labels, values, color="#3b82f6")
        axes[1].set_title("Outcome Counts")
        axes[1].set_ylabel("Episodes")
        axes[1].tick_params(axis="x", rotation=25)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No completed episodes", ha="center", va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_summary_table_page(pdf: PdfPages, *, summary: Mapping[str, Any]) -> None:
    metric_stats = summary.get("metric_stats", {})
    if not metric_stats:
        _render_message_page(
            pdf,
            title="Summary Metrics",
            lines=["No summary metrics available."],
        )
        return

    rows = []
    for metric, stats in metric_stats.items():
        rows.append(
            [
                "final_dev (exact-known mean)" if metric == "final_dev" else metric,
                f"{safe_float(stats.get('mean')):.2f}",
                f"{safe_float(stats.get('std')):.2f}",
                f"{safe_float(stats.get('min')):.2f}",
                f"{safe_float(stats.get('max')):.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Metric Summary", fontsize=16, loc="left")
    coverage = summary.get("final_timing_target_coverage", {})
    ax.text(
        0,
        0.93,
        f"Final timing targets: exact {coverage.get('exact_count', 0)}/{coverage.get('target_count', 0)}, interval {coverage.get('interval_count', 0)}, unknown {coverage.get('unknown_count', 0)}",
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Mean", "Std", "Min", "Max"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_distribution_page(pdf: PdfPages, episode_df: pd.DataFrame) -> None:
    normalized = normalize_episode_df(episode_df)
    metrics = [
        ("cum_reward", "Cumulative Reward"),
        ("final_dev", "Final Exact-known Deviation / AC"),
        ("final_infr", "Final Infringement"),
        ("steps", "Episode Steps"),
        ("nonzero_actions", "Non-zero Actions"),
    ]
    available = [(column, title) for column, title in metrics if column in normalized]
    if not available:
        _render_message_page(
            pdf,
            title="Distributions",
            lines=["No distribution metrics available."],
        )
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for ax, (column, title) in zip(axes_flat, available, strict=False):
        series = normalized[column].dropna()
        if series.empty:
            ax.axis("off")
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        ax.hist(series, bins=min(12, max(4, len(series))), color="#2563eb", alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    for ax in axes_flat[len(available) :]:
        ax.axis("off")

    fig.suptitle("Episode Distributions", fontsize=18, x=0.05, ha="left")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_correlation_page(pdf: PdfPages, episode_df: pd.DataFrame) -> None:
    normalized = normalize_episode_df(episode_df)
    required_pairs = [
        ("nonzero_actions", "cum_reward", "Actions vs Reward"),
        ("nonzero_actions", "final_dev", "Actions vs Exact-known Deviation"),
        ("nonzero_actions", "final_infr", "Actions vs Final Infringement"),
        ("init_dev", "cum_reward", "Initial Deviation vs Reward"),
        ("final_dev", "cum_reward", "Final Exact-known Deviation vs Reward"),
        ("final_infr", "cum_reward", "Final Infringement vs Reward"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Correlations and Relationships", fontsize=18, x=0.05, ha="left")
    axes_flat = axes.flatten()

    plot_count = 0
    for x_col, y_col, title in required_pairs:
        if x_col not in normalized or y_col not in normalized:
            continue
        pair = normalized[[x_col, y_col]].dropna()
        if len(pair) < 2:
            continue
        ax = axes_flat[plot_count]
        ax.scatter(pair[x_col], pair[y_col], alpha=0.75, color="#0f766e")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        plot_count += 1
        if plot_count >= 8:
            break

    numeric = normalized.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if not numeric.empty and numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        ax = axes_flat[min(plot_count, len(axes_flat) - 1)]
        image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title("Numeric Correlation Matrix")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plot_count += 1

    for ax in axes_flat[plot_count:]:
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _action_reward_rows(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if "action_ids" not in step_df or "reward_total" not in step_df:
        return pd.DataFrame(rows)

    for _, row in step_df.iterrows():
        reward_total = safe_float(row.get("reward_total"))
        for action_id in _flatten_action_values(row.get("action_ids")):
            rows.append({"action_id": int(action_id), "reward_total": reward_total})
    return pd.DataFrame(rows)


def _render_action_analysis_page(pdf: PdfPages, step_df: pd.DataFrame) -> None:
    normalized = normalize_step_df(step_df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Action Analysis", fontsize=18, x=0.05, ha="left")

    action_rows = _action_reward_rows(normalized)
    if not action_rows.empty:
        counts = action_rows["action_id"].value_counts().sort_index()
        axes[0].bar(counts.index.astype(str), counts.values, color="#7c3aed")
        axes[0].set_title("Action Frequency")
        axes[0].set_xlabel("Action Id")

        avg_reward = action_rows.groupby("action_id")["reward_total"].mean().sort_index()
        axes[1].bar(avg_reward.index.astype(str), avg_reward.values, color="#0891b2")
        axes[1].set_title("Avg Step Reward by Action")
        axes[1].set_xlabel("Action Id")
        axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No action ids available", ha="center", va="center")
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No action-reward data available", ha="center", va="center")

    if {"step", "episode_idx", "nonzero_action_count_step"} <= set(normalized.columns):
        progress = normalized[["episode_idx", "step", "nonzero_action_count_step"]].dropna()
        if not progress.empty:
            max_step_by_episode = (
                progress.groupby("episode_idx")["step"].transform("max").clip(lower=1)
            )
            progress = progress.assign(
                normalized_step=progress["step"] / max_step_by_episode,
            )
            nonzero_progress = progress[progress["nonzero_action_count_step"] > 0]
            axes[2].hist(
                nonzero_progress["normalized_step"],
                bins=10,
                range=(0.0, 1.0),
                color="#16a34a",
                alpha=0.85,
            )
            axes[2].set_title("Non-zero Action Timing")
            axes[2].set_xlabel("Normalized Episode Progress")
        else:
            axes[2].axis("off")
            axes[2].text(0.5, 0.5, "No action timing data", ha="center", va="center")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "No action timing data", ha="center", va="center")

    for ax in axes:
        ax.grid(alpha=0.2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_warning_page(pdf: PdfPages, episode_df: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for _, row in episode_df.iterrows():
        flags = row.get("warning_flags", [])
        if not flags:
            continue
        rows.append(
            [
                str(safe_int(row.get("episode_idx"), default=-1)),
                str(row.get("seed", "?")),
                str(row.get("outcome", "?")),
                ", ".join(str(flag) for flag in flags),
            ]
        )

    if not rows:
        _render_message_page(
            pdf,
            title="Potential Problems",
            lines=["No warning flags were triggered."],
        )
        return

    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.axis("off")
    ax.set_title("Potential Problems", fontsize=16, loc="left")
    table = ax.table(
        cellText=rows[:18],
        colLabels=["Episode", "Seed", "Outcome", "Flags"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_timeline_pages(pdf: PdfPages, episode_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    selected = select_representative_episodes(episode_df)
    if not selected:
        _render_message_page(
            pdf,
            title="Episode Timelines",
            lines=["No representative episodes available."],
        )
        return

    normalized_steps = normalize_step_df(step_df)
    for label, episode_idx in selected.items():
        episode_steps = normalized_steps[normalized_steps["episode_idx"] == episode_idx].copy()
        if episode_steps.empty:
            continue

        episode_steps = episode_steps.sort_values("step")
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(
            f"Episode Timeline: {label} (episode {episode_idx})",
            fontsize=16,
            x=0.05,
            ha="left",
        )

        axes[0].plot(episode_steps["step"], episode_steps["cumulative_reward"], color="#1d4ed8")
        axes[0].set_ylabel("Cum Reward")
        axes[0].grid(alpha=0.2)

        axes[1].plot(
            episode_steps["step"],
            episode_steps["reward_deviation"],
            label="deviation",
            color="#0891b2",
        )
        axes[1].plot(
            episode_steps["step"],
            episode_steps["reward_conflict"],
            label="conflict",
            color="#dc2626",
        )
        axes[1].plot(
            episode_steps["step"],
            episode_steps["reward_action"],
            label="action",
            color="#7c3aed",
        )
        if "reward_bonus" in episode_steps:
            axes[1].plot(
                episode_steps["step"],
                episode_steps["reward_bonus"],
                label="bonus",
                color="#16a34a",
            )
        axes[1].set_ylabel("Reward Parts")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.2)

        axes[2].plot(
            episode_steps["step"],
            episode_steps["current_deviation_per_ac"],
            color="#ea580c",
        )
        axes[2].set_ylabel("Deviation / AC")
        axes[2].grid(alpha=0.2)

        axes[3].plot(
            episode_steps["step"],
            episode_steps["current_infringement_total"],
            label="infringement",
            color="#991b1b",
        )
        if "nonzero_action_count_step" in episode_steps:
            axes[3].bar(
                episode_steps["step"],
                episode_steps["nonzero_action_count_step"],
                alpha=0.25,
                color="#4b5563",
                label="nonzero actions",
            )
        axes[3].set_ylabel("Infr / Actions")
        axes[3].set_xlabel("Step")
        axes[3].legend(loc="upper right")
        axes[3].grid(alpha=0.2)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_pdf_report(
    *,
    output_path: Path,
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    """Write a multi-page matplotlib PDF report."""

    normalized_episodes = normalize_episode_df(episode_df)
    normalized_steps = normalize_step_df(step_df)
    summary = build_summary(normalized_episodes, normalized_steps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _render_overview_page(
            pdf,
            episode_df=normalized_episodes,
            metadata=metadata,
            summary=summary,
        )
        _render_summary_table_page(pdf, summary=summary)
        _render_distribution_page(pdf, normalized_episodes)
        _render_correlation_page(pdf, normalized_episodes)
        _render_action_analysis_page(pdf, normalized_steps)
        _render_warning_page(pdf, normalized_episodes)
        _render_timeline_pages(pdf, normalized_episodes, normalized_steps)

from collections import Counter, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from actions.rewards import get_time_deviations
from config.config import Config
from utils.infringement_utils import get_world_infringements


class ATCEpisodeMetricsCallback(BaseCallback):
    """
    SB3 callback for logging episode-level ATC metrics.

    Logs:
    - Initial delay/deviation
    - Initial infringement total
    - Final delay/deviation
    - Final infringement total
    - Rolling success rate
    - Rolling absolute delay
    - Rolling infringement total
    - Infringement reason counts
    - Action counts and action reward statistics

    Important:
    - Terminal metrics should come from info["episode_metrics"].
    - This avoids reading the wrong world after SB3 auto-resets the env.
    """

    # Every reason ATCSingleAgentEnv can end an episode with. Logged as a full 0/1 set on each
    # episode so each series is a true fraction; see the note where they are recorded.
    _END_REASONS = (
        "success",
        "delayed",
        "timeout",
        "separation",
        "airspace_exit",
        "fallback_world",
        "unknown",
    )

    def __init__(
        self,
        rolling_window: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        # Per-env episode state, sized lazily in _on_training_start once the vec env is known.
        # SB3 steps N envs in parallel; each needs its own in-progress action history and
        # episode-start flag so episodes from different envs are never interleaved into one
        # another's metrics. The rolling histories below are intentionally shared (they are a
        # rolling mean over the last `rolling_window` completed episodes regardless of source env).
        self._episode_action_history: list | None = None
        self._needs_episode_start_log: list | None = None

        self.rolling_window = int(rolling_window)

        self.success_history = deque(maxlen=self.rolling_window)
        self.delay_history = deque(maxlen=self.rolling_window)
        self.abs_delay_history = deque(maxlen=self.rolling_window)
        self.infringement_history = deque(maxlen=self.rolling_window)

        self.episode_count = 0



    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _sanitize_tb_key(value) -> str:
        """
        Convert arbitrary reason/type strings into TensorBoard-safe key parts.
        """
        safe = str(value)

        replacements = {
            " ": "_",
            "/": "_",
            "\\": "_",
            ":": "_",
            ";": "_",
            ",": "_",
            ".": "_",
            "(": "",
            ")": "",
            "[": "",
            "]": "",
            "{": "",
            "}": "",
            "'": "",
            '"': "",
        }

        for old, new in replacements.items():
            safe = safe.replace(old, new)

        return safe or "unknown"

    @staticmethod
    def _get_infringement_reason(infringement) -> str:
        """
        Fallback extractor for infringement reason/type/name.
        """
        for attr in [
            "reason",
            "type",
            "infringement_type",
            "name",
            "kind",
            "category",
        ]:
            value = getattr(infringement, attr, None)

            if value is None:
                continue

            if hasattr(value, "name"):
                return str(value.name)

            return str(value)

        return infringement.__class__.__name__

    def _get_base_env(self):
        """
        Works for normal SB3 VecEnv with one env.
        """
        if self.training_env is None:
            return None

        envs = getattr(self.training_env, "envs", None)

        if envs:
            return envs[0].unwrapped

        return getattr(self.training_env, "unwrapped", self.training_env)

    def _get_logging_world(self):
        """
        Fallback world source.

        Prefer info["episode_metrics"] whenever available.
        """
        base_env = self._get_base_env()

        if base_env is None:
            return None

        simulator = getattr(base_env, "simulator", None)

        if simulator is None:
            return None

        return getattr(simulator, "world", None)

    @staticmethod
    def _rolling_mean(values) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _extract_terminal_metrics(infos):
        """
        Extract info["episode_metrics"] from SB3 infos list.
        """
        if not infos:
            return None

        for info in infos:
            if "episode_metrics" in info:
                return info["episode_metrics"]

        return None

    # =====================================================
    # Episode lifecycle
    # =====================================================

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_action_history = [[] for _ in range(n_envs)]
        self._needs_episode_start_log = [True] * n_envs

    def _on_step(self) -> bool:
        if self._needs_episode_start_log is None:
            # Safety net in case _on_training_start was not invoked for this callback.
            self._on_training_start()

        n_envs = len(self._needs_episode_start_log)
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones")

        # ---------------------------------------------
        # Log episode-start values once per episode, per env. The base env exposes the new
        # episode's initial deviation/infringement after its auto-reset; read them via the VecEnv
        # get_attr API so this works for both DummyVecEnv and SubprocVecEnv (subprocess workers).
        # ---------------------------------------------
        if any(self._needs_episode_start_log):
            initial_dev, initial_infr = self._get_initial_metrics(n_envs)
            for i in range(n_envs):
                if self._needs_episode_start_log[i]:
                    self._log_episode_start(initial_dev[i], initial_infr[i])
                    self._needs_episode_start_log[i] = False

        # ---------------------------------------------
        # Collect step info per env
        # ---------------------------------------------
        for i in range(min(n_envs, len(infos))):
            info = infos[i]
            if isinstance(info, dict) and "action" in info:
                self._episode_action_history[i].append(info)

        # ---------------------------------------------
        # Detect per-env episode end; log each finished episode in isolation so parallel envs
        # never share an action history or terminal-metrics info.
        # ---------------------------------------------
        if dones is not None:
            for i in range(min(n_envs, len(dones))):
                if dones[i]:
                    self._log_episode_end(
                        infos=[infos[i]] if i < len(infos) else [],
                        history=self._episode_action_history[i],
                    )
                    self._episode_action_history[i] = []
                    self._needs_episode_start_log[i] = True
                    self.episode_count += 1

        return True

    def _get_initial_metrics(self, n_envs: int):
        """Per-env (initial_deviation, initial_infr_total) lists via the VecEnv get_attr API.

        VecEnv.get_attr resolves through get_wrapper_attr, which walks the wrapper stack, so it
        reaches the attribute on the unwrapped ATCSingleAgentEnv through Monitor / VecNormalize for
        both Dummy and Subproc vec envs. Falls back to zeros if the attribute is unavailable.
        """
        try:
            dev = self.training_env.get_attr("initial_deviation")
            infr = self.training_env.get_attr("initial_infr_total")
        except Exception:
            dev = [0.0] * n_envs
            infr = [0.0] * n_envs
        # initial_deviation/initial_infr_total are None before the first reset; coerce defensively.
        dev = [self._safe_float(v) for v in dev]
        infr = [self._safe_float(v) for v in infr]
        return dev, infr

    # =====================================================
    # Episode logging
    # =====================================================

    def _log_episode_start(self, initial_deviation, initial_infr_total):
        self.logger.record_mean(
            "custom_metrics/deviation_initial",
            self._safe_float(initial_deviation),
        )

        self.logger.record_mean(
            "custom_metrics/infringement_total_initial",
            self._safe_float(initial_infr_total),
        )

    def _log_episode_end(self, infos=None, history=None):
        history = history if history is not None else []

        terminal_metrics = self._extract_terminal_metrics(infos)

        self.logger.record_mean(
            "debug_metrics/episode_action_history_len",
            float(len(history)),
        )

        self.logger.record_mean(
            "debug_metrics/has_terminal_episode_metrics",
            float(terminal_metrics is not None),
        )

        # =====================================================
        # Final delay / infringement metrics
        # =====================================================

        if terminal_metrics is not None:
            final_dev = self._safe_float(
                terminal_metrics.get("deviation_final", 0.0)
            )

            final_abs_dev = self._safe_float(
                terminal_metrics.get("deviation_abs_final", abs(final_dev))
            )

            final_infr = self._safe_float(
                terminal_metrics.get("infringement_total_final", 0.0)
            )

            reason_counts = Counter(
                terminal_metrics.get("infringement_reasons", {})
            )

            # _build_episode_metrics stores the flag under "success"; older/top-level info uses
            # "is_success". Read is_success first (future-proof), then fall back to "success" so
            # the training success curve reflects real episode outcomes instead of always 0.
            success = bool(
                terminal_metrics.get(
                    "is_success", terminal_metrics.get("success", False)
                )
            )

            end_reason = terminal_metrics.get("end_reason", "unknown")

            self.logger.record_mean(
                "custom_metrics/world_missing",
                0.0,
            )

        else:
            # Fallback path. Ideally this should not be used.
            world = self._get_logging_world()

            if world is None:
                self.logger.record_mean(
                    "custom_metrics/world_missing",
                    1.0,
                )
                return

            self.logger.record_mean(
                "custom_metrics/world_missing",
                0.0,
            )

            total_dev = get_time_deviations(world)
            final_dev = total_dev[1] if total_dev else 0.0

            final_dev = self._safe_float(final_dev)
            final_abs_dev = abs(final_dev)

            final_infr = 0.0
            reason_counts = Counter()

            infr_map = get_world_infringements(world)
            # Count only REALISED losses of separation (<= 3 NM and <= 500 ft).
            threshold = float(Config.LOS_SEVERITY_THRESHOLD) - 1e-9

            for infringement_list in infr_map.values():
                for infringement in infringement_list:
                    value = self._safe_float(
                        getattr(infringement, "value", 0.0)
                    )

                    if value >= threshold:
                        final_infr += value
                        reason = self._get_infringement_reason(infringement)
                        reason_counts[reason] += 1

            success = self._infer_success(
                infos=infos,
                final_abs_dev=final_abs_dev,
                final_infr=final_infr,
            )

            end_reason = "fallback_world"

        # =====================================================
        # Rolling histories
        # =====================================================

        self.delay_history.append(final_dev)
        self.abs_delay_history.append(final_abs_dev)
        self.infringement_history.append(final_infr)
        self.success_history.append(float(success))

        # Windowed-env per-flight metrics (WindowedATCEnv only).
        if terminal_metrics is not None:
            for key in ("window_on_time_rate", "window_n_landed", "window_overflow_frac",
                        "window_aman_order_rate", "window_sequence_swaps", "window_clean_stretch_rate",
                        "window_clearances", "window_again_picks", "window_multi_pick_steps"):
                if key in terminal_metrics:
                    self.logger.record_mean(
                        f"window_metrics/{key[len('window_'):]}",
                        self._safe_float(terminal_metrics[key]),
                    )

        # =====================================================
        # Log final delay / deviation
        # =====================================================

        self.logger.record_mean(
            "custom_metrics/deviation_final",
            float(final_dev),
        )

        self.logger.record_mean(
            "custom_metrics/deviation_abs_final",
            float(final_abs_dev),
        )

        self.logger.record(
            "custom_metrics/deviation_rolling",
            self._rolling_mean(self.delay_history),
        )

        self.logger.record(
            "custom_metrics/deviation_abs_rolling",
            self._rolling_mean(self.abs_delay_history),
        )

        # =====================================================
        # Log final infringement metrics
        # =====================================================

        self.logger.record_mean(
            "custom_metrics/infringement_total_final",
            float(final_infr),
        )

        self.logger.record(
            "custom_metrics/infringement_total_rolling",
            self._rolling_mean(self.infringement_history),
        )

        self.logger.record_mean(
            "custom_metrics/infringement_count_final",
            float(sum(reason_counts.values())),
        )

        for reason, count in reason_counts.items():
            safe_reason = self._sanitize_tb_key(reason)

            self.logger.record_mean(
                f"custom_metrics/infr/{safe_reason}",
                float(count),
            )

        if not reason_counts:
            self.logger.record_mean(
                "custom_metrics/infr/none",
                1.0,
            )

        # =====================================================
        # Log success
        # =====================================================

        self.logger.record_mean(
            "custom_metrics/success",
            float(success),
        )

        self.logger.record(
            "custom_metrics/success_rolling",
            self._rolling_mean(self.success_history),
        )

        # =====================================================
        # Log end reason
        # =====================================================

        # Record 0.0 for every reason that did NOT fire, so record_mean() yields the FRACTION
        # of episodes ending each way. Previously only the reason that fired was recorded, and
        # only ever as 1.0 -- so every key that appeared at all averaged to exactly 1.0 and the
        # series carried no frequency information. `end_reason/separation` read 1 whether one
        # episode in a thousand busted or every single one did, which made it useless as the
        # safety diagnostic it was being relied on for.
        safe_end_reason = self._sanitize_tb_key(end_reason)

        for known in self._END_REASONS:
            self.logger.record_mean(
                f"custom_metrics/end_reason/{known}",
                1.0 if known == safe_end_reason else 0.0,
            )
        if safe_end_reason not in self._END_REASONS:
            # Unanticipated reason: still log it, accepting that its series is presence-only
            # until it is added to _END_REASONS.
            self.logger.record_mean(f"custom_metrics/end_reason/{safe_end_reason}", 1.0)

        # =====================================================
        # Action stats
        # =====================================================

        counts = [0] * int(Config.NUM_ACTIONS)
        rewards = [0.0] * int(Config.NUM_ACTIONS)

        bonus = 0.0
        dev = 0.0
        infr = 0.0

        for h in history:
            act = h.get("action", -1)

            try:
                act = int(act)
            except Exception:
                act = -1

            if 0 <= act < int(Config.NUM_ACTIONS):
                counts[act] += 1
                rewards[act] += self._safe_float(h.get("reward", 0.0))

            rp = h.get("reward_parts", None)

            if rp is not None:
                bonus += self._safe_float(getattr(rp, "bonus", 0.0))
                dev += self._safe_float(getattr(rp, "deviation", 0.0))
                infr += self._safe_float(getattr(rp, "conflict", 0.0))

        nonzero = sum(counts[1:])

        self.logger.record_mean(
            "custom_metrics/nonzero_action",
            float(nonzero),
        )

        for i, c in enumerate(counts):
            self.logger.record_mean(
                f"custom_metrics/action_{i}_count",
                float(c),
            )

            avg_r = rewards[i] / c if c > 0 else 0.0

            self.logger.record_mean(
                f"custom_metrics/action_{i}_reward",
                float(avg_r),
            )

        if len(history) > 0:
            denom = float(len(history))

            self.logger.record_mean(
                "custom_metrics/bonus_reward",
                bonus / denom,
            )

            self.logger.record_mean(
                "custom_metrics/infringement_reward",
                infr / denom,
            )

            self.logger.record_mean(
                "custom_metrics/time_deviation_reward",
                dev / denom,
            )

        else:
            self.logger.record_mean(
                "custom_metrics/bonus_reward",
                0.0,
            )

            self.logger.record_mean(
                "custom_metrics/infringement_reward",
                0.0,
            )

            self.logger.record_mean(
                "custom_metrics/time_deviation_reward",
                0.0,
            )

    def _infer_success(
        self,
        infos,
        final_abs_dev: float,
        final_infr: float,
    ) -> bool:
        """
        Infer success for an episode.

        Priority:
        1. Use info['is_success'] if env provides it.
        2. Use info['success'] if env provides it.
        3. Otherwise define success from final delay and infringement.
        """

        if infos:
            for info in infos:
                if "is_success" in info:
                    return bool(info["is_success"])

        delay_threshold = float(
            getattr(Config, "SUCCESS_DELAY_THRESHOLD", 60.0)
        )

        return (
            final_abs_dev <= delay_threshold
            and final_infr <= 0.0
        )
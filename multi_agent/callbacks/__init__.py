"""Fixed training callback registry."""

from callbacks.metrics_callback import ATCEpisodeMetricsCallback


def build_training_callbacks():
    return [ATCEpisodeMetricsCallback]

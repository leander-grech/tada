"""Explicit procedure selection for benchmark inputs."""

from training.use_case import use_case

DENSITIES = (10, 15, 20)


def procedure_inputs(campaign="bgy"):
    return use_case(campaign).inputs()


def procedure_root(campaign="bgy"):
    case = use_case(campaign)
    case.inputs()
    return case.config_root


def procedure_selection(campaign, count):
    """The compatibility extension; absent for the BGY ten-aircraft default."""
    if campaign == "bgy" and count == 10:
        return None
    return {
        "version": 1,
        "campaign": campaign,
        "aircraft_count": count,
        "purpose": "primary" if count == 10 else "density-diagnostic",
    }


def selection(compatibility):
    extension = compatibility.get("campaign_selection")
    if extension is None:
        return "bgy", 10
    if (
        set(extension) != {"version", "campaign", "aircraft_count", "purpose"}
        or extension["version"] != 1
        or extension["campaign"] not in {"mxp", "bgy"}
    ):
        raise ValueError("unsupported campaign selection")
    count = extension["aircraft_count"]
    if (
        type(count) is not int
        or count not in DENSITIES
        or extension != procedure_selection(extension["campaign"], count)
    ):
        raise ValueError("unsupported campaign density")
    return extension["campaign"], count

"""Local support-readiness experiment, not a calibrated grasp-success detector.

The right hand keeps its orientation and finger commands during insertion.
Compare reported actuator effort with that hand's earlier unloaded baseline.
Both ring-finger channels must stay above their experimental deltas for a short
history. Other contacts can also create this pattern: RGB and a test lift remain
necessary to verify a grasp. These thresholds have not been calibrated natively.
"""

import math
import statistics

MINIMUM_DELTA = {"rh_ring_mcp_pitch": 0.03, "rh_ring_dip": 0.01}


def left_blocking_loss(baseline, latest, targets):
    """Flag two formerly blocked fingers becoming almost free at the latest sample.

    This is an experimental withdrawal hint, not contact identity or grasp proof.
    Effort and command error are correlated by the native PD law.
    """
    if len(baseline) < 5:
        return dict(
            lost_hint=False, reference_usable=False, reason="insufficient_reference"
        )
    channels = {}
    for name in ("lh_thumb_cmc_pitch", "lh_index_mcp_pitch"):
        positions = [r.get("position", {}).get(name) for r in baseline]
        efforts = [r.get("servo_effort", {}).get(name) for r in baseline]
        current = [
            latest.get(field, {}).get(name)
            for field in ("position", "velocity", "effort")
        ]
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in [*positions, *efforts, *current]
        ):
            return dict(
                lost_hint=False,
                reference_usable=False,
                reason="missing_or_nonfinite_feedback",
            )
        errors = [targets[name] - v for v in positions]
        blocked = statistics.median(errors)
        load = statistics.median(abs(v) for v in efforts)
        error = abs(targets[name] - current[0])
        usable = blocked >= 0.015 and load >= 0.1
        channels[name] = dict(
            reference_error_rad=blocked,
            reference_effort=load,
            latest_error_rad=error,
            latest_effort=current[2],
            latest_velocity_rad_s=current[1],
            reference_usable=usable,
            released=usable
            and error <= 0.2 * blocked
            and abs(current[2]) <= 0.2 * load
            and abs(current[1]) <= 0.03,
        )
    usable = all(c["reference_usable"] for c in channels.values())
    lost = all(c["released"] for c in channels.values())
    return dict(
        lost_hint=lost,
        reference_usable=usable,
        channels=channels,
        reason="blocked_to_free" if lost else "no_blocked_to_free_pattern",
        scope="Latest native snapshot compared with earlier closed reference; PD error and effort are not independent tactile measurements",
    )


def support_evidence(baseline, recent):
    """Read only timestamps and actuator efforts; never object/contact outputs."""
    if len(baseline) < 5 or len(recent) < 5:
        return {"ready": False, "reason": "insufficient_history"}
    stamps = [row["seconds"] for row in recent]
    earlier = [row["seconds"] for row in baseline]
    if (
        not all(math.isfinite(t) for t in stamps + earlier)
        or any(b <= a for a, b in zip(stamps, stamps[1:]))
        or stamps[-1] - stamps[0] < 0.16 - 1e-9
        or max(earlier) >= stamps[0]
    ):
        return {"ready": False, "reason": "insufficient_advancing_history"}
    delta = {}
    for name, threshold in MINIMUM_DELTA.items():
        unloaded = [row["servo_effort"].get(name, float("nan")) for row in baseline]
        observed = [row["servo_effort"].get(name, float("nan")) for row in recent]
        if not all(math.isfinite(v) for v in unloaded + observed):
            return {"ready": False, "reason": "missing_or_nonfinite_effort"}
        reference = statistics.median(unloaded)
        values = [v - reference for v in observed]
        delta[name] = {
            "baseline": reference,
            "minimum_delta": min(values),
            "median_delta": statistics.median(values),
            "threshold": threshold,
        }
    ready = all(v["minimum_delta"] >= v["threshold"] for v in delta.values())
    return {
        "ready": ready,
        "reason": "support_pattern" if ready else "no_sustained_support_pattern",
        "channels": delta,
    }

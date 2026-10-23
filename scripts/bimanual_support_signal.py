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

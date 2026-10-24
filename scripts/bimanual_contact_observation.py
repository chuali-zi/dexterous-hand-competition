"""Retain contact observations for offline diagnosis; no motion commands."""

import base64
from bisect import bisect_right
import json
import math
from pathlib import Path
import subprocess
import sys

from bimanual_contact_vision import check_tilt_visibility
from rabokit_bricks import detect
from rabokit_camera import png_bytes
from hand_competition.closed_loop.artifacts import read_ppm

CHANNELS = (
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
)


def stamp_seconds(row):
    return row["stamp"][0] + row["stamp"][1] * 1e-9


def compact_hand(row):
    return dict(
        stamp=row["stamp"],
        wall_seconds=row.get("wall_seconds"),
        **{
            field: {
                n: (
                    row[field].get(n)
                    if isinstance(row[field].get(n), (int, float))
                    and math.isfinite(row[field][n])
                    else None
                )
                for n in CHANNELS
            }
            for field in ("position", "velocity", "effort")
        },
    )


def side_preview(width, height, pixels, components):
    visible = [c for c in components if c["area"] >= 10]
    if not visible:
        return None
    x0 = max(0, min(c["bounds"][0] for c in visible) - 40)
    x1 = min(width, max(c["bounds"][1] for c in visible) + 41)
    y0 = max(0, min(c["bounds"][2] for c in visible) - 40)
    y1 = min(height, max(c["bounds"][3] for c in visible) + 41)
    crop = b"".join(
        pixels[(y * width + x0) * 3 : (y * width + x1) * 3] for y in range(y0, y1)
    )
    return dict(
        camera="base_cam3",
        bounds_xyxy=[x0, y0, x1, y1],
        png_base64=base64.b64encode(png_bytes(x1 - x0, y1 - y0, crop)).decode(),
    )


def closing_rgb_evidence(root, config):
    trace = json.loads((root / "contact-close.json").read_text())
    rgb = trace.get("rgb_trace", {})
    if rgb.get("error"):
        return dict(error=rgb["error"])
    frames = json.loads((root / "contact_closed/robot-frames.json").read_text())[
        "frames"
    ]
    joints = trace["joint_samples"]
    times = [stamp_seconds(r) for r in joints]
    rows = []
    for frame in rgb.get("frames", []):
        folder = root / "contact-close-rgb" / frame["directory"]
        metadata, green = {}, {}
        for name in ("base_cam", "base_cam2"):
            metadata[name] = json.loads((folder / name / "camera.json").read_text())
            w, h, pixels = read_ppm(folder / name / "frame.ppm")
            found = detect(w, h, pixels)
            green[name] = dict(
                components=[
                    c
                    for c in found["candidates"] + found["rejected_components"]
                    if c["color"] == "green"
                ]
            )
        checked = check_tilt_visibility(frames, metadata, green, config)
        seconds = stamp_seconds(frame)
        index = bisect_right(times, seconds) - 1
        gap = seconds - times[index] if index >= 0 else None
        rows.append(
            dict(
                stamp=frame["stamp"],
                relative_command_seconds=frame["relative_command_seconds"],
                excluded=checked["excluded"],
                cameras={
                    n: dict(
                        available=c["available"],
                        excluded=c["excluded"],
                        maximum_bound_violation_px=c.get("maximum_bound_violation_px"),
                        observed_bounds=c.get("observed_bounds"),
                    )
                    for n, c in checked["cameras"].items()
                },
                hand_sample_gap_seconds=gap,
                hand=(
                    compact_hand(joints[index])
                    if gap is not None and gap <= 0.04
                    else None
                ),
            )
        )
    return dict(
        rows=rows,
        source="Matched stereo during closure, fixed camera TF from the closed observation; preceding hand sample only",
    )


def start_tilt_recording(root):
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("rabokit_robot_frames.py")),
            "--hand-trace",
            "--seconds",
            "30",
            "--output",
            str(root / "tilt-trace.json"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def finish_tilt_recording(process, root, until=None):
    if process is None:
        return None
    try:
        stdout, stderr = process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=1)
        (root / "tilt-trace.log").write_text(stdout + stderr)
        return dict(
            error="Recorder did not finish before recovery; terminated without extending observation"
        )
    (root / "tilt-trace.log").write_text(stdout + stderr)
    if process.returncode:
        return dict(error=f"Recorder exited {process.returncode}")
    trace = json.loads((root / "tilt-trace.json").read_text())
    retained = []
    for row in trace["hand_joint_states"]:
        seconds = stamp_seconds(row)
        if until is not None and seconds > until:
            break
        if not retained or seconds - stamp_seconds(retained[-1]) >= 0.2 - 1e-9:
            retained.append(compact_hand(row))
    return dict(
        samples=retained,
        started_monotonic=trace["started_monotonic"],
        scope="Native hand samples at at least 0.2s spacing, ending at the latest tilted native snapshot when available; coverage is recorded, not assumed",
    )

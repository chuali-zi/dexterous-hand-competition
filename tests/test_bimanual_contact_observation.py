import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bimanual_contact_observation import (
    closing_rgb_evidence,
    finish_tilt_recording,
    side_preview,
)


class ContactObservationTests(unittest.TestCase):
    def test_side_preview_keeps_native_pixel_scale_and_clips_at_image_border(self):
        rgb = bytes([20, 200, 20]) * 110 * 90
        preview = side_preview(110, 90, rgb, [dict(area=25, bounds=[105, 109, 85, 89])])
        self.assertEqual(preview["bounds_xyxy"], [65, 45, 110, 90])
        png = base64.b64decode(preview["png_base64"])
        self.assertEqual(struct.unpack("!II", png[16:24]), (45, 45))
        self.assertIsNone(side_preview(110, 90, rgb, []))

    def test_closing_rgb_uses_preceding_hand_sample_and_never_the_future_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "contact_closed").mkdir()
            (root / "contact_closed/robot-frames.json").write_text(
                json.dumps(dict(frames={}))
            )
            folder = root / "contact-close-rgb/frame"
            for name in ("base_cam", "base_cam2"):
                (folder / name).mkdir(parents=True)
                (folder / name / "camera.json").write_text("{}")

            def row(nano, value):
                return dict(
                    stamp=[1, nano],
                    position=dict(lh_thumb_ip=value),
                    velocity={},
                    effort={},
                )

            trace = dict(
                joint_samples=[row(180000000, 1.1), row(220000000, 9.9)],
                rgb_trace=dict(
                    frames=[
                        dict(
                            stamp=[1, 200000000],
                            relative_command_seconds=0.2,
                            directory="frame",
                        )
                    ]
                ),
            )
            (root / "contact-close.json").write_text(json.dumps(trace))
            with patch(
                "bimanual_contact_observation.read_ppm", return_value=(1, 1, b"\0\0\0")
            ), patch(
                "bimanual_contact_observation.detect",
                return_value=dict(candidates=[], rejected_components=[]),
            ), patch(
                "bimanual_contact_observation.check_tilt_visibility",
                return_value=dict(excluded=False, cameras={}),
            ):
                result = closing_rgb_evidence(root, {})
            self.assertAlmostEqual(result["rows"][0]["hand_sample_gap_seconds"], 0.02)
            self.assertEqual(result["rows"][0]["hand"]["position"]["lh_thumb_ip"], 1.1)

    def test_finished_tilt_trace_excludes_return_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = [
                dict(stamp=[1, i * 100000000], position={}, velocity={}, effort={})
                for i in range(6)
            ]
            (root / "tilt-trace.json").write_text(
                json.dumps(dict(hand_joint_states=samples, started_monotonic=4))
            )
            proc = SimpleNamespace(
                communicate=Mock(return_value=("done", "")), returncode=0
            )
            result = finish_tilt_recording(proc, root, until=1.3)
            self.assertEqual(
                [r["stamp"] for r in result["samples"]], [[1, 0], [1, 200000000]]
            )

    def test_unfinished_recorder_is_terminated_without_waiting_for_the_full_window(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            proc = Mock()
            proc.communicate.side_effect = [
                subprocess.TimeoutExpired("recorder", 1),
                ("", ""),
            ]
            result = finish_tilt_recording(proc, Path(temp))
            proc.terminate.assert_called_once()
            self.assertIn("terminated", result["error"])


if __name__ == "__main__":
    unittest.main()

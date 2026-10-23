from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bimanual_contact_vision import bound_evidence, check_tilt_visibility
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT


class ContactVisionTests(unittest.TestCase):
    def test_partial_non_square_fragment_does_not_need_full_brick_reconstruction(self):
        fragment = dict(area=20, bounds=[11, 14, 12, 18])
        result = bound_evidence([[10, 10], [20, 20]], [fragment])
        self.assertTrue(result["available"])
        self.assertFalse(result["excluded"])
        self.assertFalse(bound_evidence([[10, 10], [20, 20]], [])["available"])

    def test_occlusion_cannot_explain_green_outside_the_full_projected_box(self):
        result = bound_evidence(
            [[10, 10], [20, 20]], [dict(area=20, bounds=[12, 14, 1, 5])]
        )
        self.assertTrue(result["excluded"])
        self.assertEqual(result["maximum_bound_violation_px"], 9)

    def test_native63_green_bounds_exclude_the_intended_region_in_two_cameras(self):
        config = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/bimanual-contact.json"
            ).read_text()
        )
        frames = {
            LEFT_ROOT: dict(
                parent="world", xyz=[-0.13, 0.004, 1.255], xyzw=[0, 0, 0, 1]
            ),
            RIGHT_ROOT: dict(
                parent="world", xyz=[-0.13, -0.004, 1.255], xyzw=[0, 0, 0, 1]
            ),
        }
        metadata, observations = {}, {}
        for name, x, bounds in [
            ("base_cam", 0.39, [84, 128, 197, 220]),
            ("base_cam2", -0.16, [525, 563, 197, 219]),
        ]:
            frames[name] = dict(parent="world", xyz=[x, 0.1, 2.6515], xyzw=[1, 0, 0, 0])
            metadata[name] = dict(
                frame_id=name,
                k=[1515.3003635220105, 0, 320, 0, 1515.3003635220105, 240, 0, 0, 1],
            )
            observations[name] = dict(components=[dict(area=346, bounds=bounds)])
        result = check_tilt_visibility(frames, metadata, observations, config)
        self.assertTrue(result["excluded"])
        observations["base_cam2"]["components"] = []
        self.assertFalse(
            check_tilt_visibility(frames, metadata, observations, config)["excluded"]
        )


if __name__ == "__main__":
    unittest.main()

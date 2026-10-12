"""Offline arithmetic and evidence-boundary regression tests, not robot tests."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hand_competition.evaluation import append_review, evaluate, load_record

ROOT = Path(__file__).resolve().parents[1]


def brick_record(count=10, shape=False, color=False):
    record = load_record(ROOT / 'examples/brick-final-state.json')
    record['final_state'] = [
        {'id': f'b{i}', 'released': True, 'stable_for_s': 3,
         'stable_valid': True, 'shape_awarded': shape, 'color_awarded': color}
        for i in range(count)
    ]
    return record


class EvaluationTests(unittest.TestCase):
    def test_full_brick_scores_and_coefficients(self):
        record = brick_record(shape=True, color=True)
        self.assertEqual(evaluate(record)['raw_score'], 50)
        record['control_mode'] = 'teleoperation'
        self.assertEqual(evaluate(record)['weighted_score'], 25)
        record = brick_record()
        self.assertEqual(evaluate(record)['weighted_score'], 30)

    def test_wrong_color_preserves_base_and_shape(self):
        result = evaluate(brick_record(shape=True, color=False))
        self.assertEqual(result['components'], {'stable_base': 30, 'shape': 10, 'color': 0})
        self.assertEqual(result['raw_score'], 40)

    def test_color_requires_shape_under_verified_rule(self):
        record = brick_record(1, shape=False, color=True)
        with self.assertRaisesRegex(ValueError, "color points require"):
            evaluate(record)
        record["final_state"][0]["shape_awarded"] = True
        self.assertEqual(evaluate(record)["raw_score"], 5)

    def test_collapse_uses_final_state_not_past_success(self):
        original = brick_record(shape=True, color=True)
        final = copy.deepcopy(original)
        for item in final['final_state'][6:]:
            item.update(stable_valid=False, stable_for_s=0,
                        shape_awarded=False, color_awarded=False)
        self.assertEqual(evaluate(original)['raw_score'], 50)
        self.assertEqual(evaluate(final)['raw_score'], 30)

    def test_stability_boundary_requires_release_and_three_seconds(self):
        for duration in (0, 2.999):
            record = brick_record(1)
            record['final_state'][0]['stable_for_s'] = duration
            with self.assertRaises(ValueError):
                evaluate(record)
        record = brick_record(1)
        self.assertEqual(evaluate(record)['raw_score'], 3)
        record['final_state'][0]['released'] = False
        with self.assertRaises(ValueError):
            evaluate(record)

    def test_unstable_block_cannot_retain_awards(self):
        record = brick_record(1, shape=True)
        record['final_state'][0]['stable_valid'] = False
        with self.assertRaises(ValueError):
            evaluate(record)

    def test_duplicate_ids_and_inventory_limit(self):
        record = brick_record(2)
        record['final_state'][1]['id'] = record['final_state'][0]['id']
        with self.assertRaises(ValueError):
            evaluate(record)
        with self.assertRaises(ValueError):
            evaluate(brick_record(11))

    def test_full_assembly_and_contact_is_not_insertion(self):
        record = load_record(ROOT / 'examples/assembly-final-state.json')
        self.assertEqual(evaluate(record)['weighted_score'], 25)
        for item in record['final_state']:
            item['effective_inserted'] = False
        self.assertEqual(evaluate(record)['raw_score'], 0)
        record['final_state'][0]['effective_inserted'] = True
        record['final_state'][4]['effective_inserted'] = True
        self.assertEqual(evaluate(record)['raw_score'], 17)

    def test_assembly_inventory_and_unknown_kind_rejected(self):
        record = load_record(ROOT / 'examples/assembly-final-state.json')
        record['final_state'][4]['kind'] = 'preset'
        with self.assertRaises(ValueError):
            evaluate(record)
        record['final_state'][4]['kind'] = 'unknown'
        with self.assertRaises(ValueError):
            evaluate(record)

    def test_empty_snapshot_scores_zero(self):
        self.assertEqual(evaluate(brick_record(0))['raw_score'], 0)

    def test_nonfinite_negative_boolean_or_overtime_duration_rejected(self):
        for value in (float('nan'), float('inf'), 10**400, -1, True, '240', 300.001):
            record = brick_record(1)
            record['elapsed_s'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate(record)
        record = brick_record(1)
        record['elapsed_s'] = 300
        self.assertEqual(evaluate(record)['raw_score'], 3)

    def test_stability_cannot_exceed_elapsed(self):
        record = brick_record(1)
        record['final_state'][0]['stable_for_s'] = 241
        with self.assertRaises(ValueError):
            evaluate(record)

    def test_mode_task_evidence_and_version_fail_closed(self):
        for key, value in [('control_mode', 'mixed'), ('task', 'practice'),
                           ('evidence_kind', 'official_score'), ('schema_version', True),
                           ('schema_version', 2), ('evidence_ref', '')]:
            record = brick_record(1)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate(record)

    def test_boolean_fields_not_truthy_strings_or_integers(self):
        for value in ('false', 1, None):
            record = brick_record(1)
            record['final_state'][0]['stable_valid'] = value
            with self.assertRaises(ValueError):
                evaluate(record)

    def test_unknown_missing_and_wrong_structure_rejected(self):
        for change in ('unknown', 'missing', 'wrong_state', 'wrong_item'):
            record = brick_record(1)
            if change == 'unknown':
                record['ground_truth_pose'] = [0, 0, 0]
            elif change == 'missing':
                del record['evidence_ref']
            elif change == 'wrong_state':
                record['final_state'] = {}
            else:
                record['final_state'] = [42]
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate(record)

    def test_json_duplicate_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            for content in ('{"task":"brick","task":"assembly"}', '{"elapsed_s":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_record(path)

    def test_log_preserves_records_and_marks_nonofficial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reviews.jsonl'
            record = brick_record(1)
            before = copy.deepcopy(record)
            append_review(path, record)
            append_review(path, record)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(record, before)
            self.assertEqual(lines[0]['record'], record)
            self.assertEqual(lines[0]['record_sha256'], lines[1]['record_sha256'])
            self.assertFalse(lines[0]['result']['official_judge'])
            record['control_mode'] = 'mixed'
            with self.assertRaises(ValueError):
                append_review(path, record)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_cli_examples_are_synthetic(self):
        for name, score in [('brick', 35), ('assembly', 25)]:
            run = subprocess.run([sys.executable, '-m', 'hand_competition',
                                  str(ROOT / f'examples/{name}-final-state.json')],
                                 cwd=ROOT, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout)
            self.assertEqual(result['weighted_score'], score)
            self.assertEqual(result['evidence_kind'], 'synthetic_example')
            self.assertFalse(result['official_judge'])

    def test_cli_invalid_file_fails_without_score(self):
        run = subprocess.run([sys.executable, '-m', 'hand_competition', '/nonexistent/record.json'],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, '')


if __name__ == '__main__':
    unittest.main()

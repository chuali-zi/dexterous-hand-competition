#!/usr/bin/env python3
"""Offline analytic geometry fixtures + RGB noise experiments, no robot/SDK."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_competition.closed_loop.experiments import run_trial, replay, code_identity
from hand_competition.closed_loop.geometry import Camera, triangulate
from hand_competition.closed_loop.synthetic import SensorNoise


def analytic_scan():
    """Pixel fixture generator does NOT call production Camera.project/ray.

    These object coordinates are evaluator-only synthetic ground truth, never
    provided to the controller or used to claim a real calibration measurement.
    """
    rotation = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    nominal = (Camera('left', (-0.1, 0, 0), rotation, fx=100, fy=100),
               Camera('right', (0.1, 0, 0), rotation, fx=100, fy=100))
    cases = []
    for jitter in (0.0, 0.5, 1.0, 2.0):
        for shift in (0.0, 0.005, 0.02):
            rng = random.Random(2026)
            cameras = tuple(replace(c, origin=(c.origin[0]+shift, 0, 0)) for c in nominal)
            errors = []; residuals = []; rejected = 0
            for x in (-0.15, 0, 0.15):
                for y in (-0.06, 0.06):
                    for z in (0.6, 1.0, 1.4):
                        # Independent closed-form pinhole equations, base z-forward.
                        first = (80+100*(x+0.1)/z+rng.uniform(-jitter, jitter),
                                 60+100*y/z+rng.uniform(-jitter, jitter))
                        second = (80+100*(x-0.1)/z+rng.uniform(-jitter, jitter),
                                  60+100*y/z+rng.uniform(-jitter, jitter))
                        try: estimated, residual = triangulate(cameras[0], first, cameras[1], second)
                        except ValueError: rejected += 1; continue
                        errors.append(math.sqrt(sum((estimated[i]-(x, y, z)[i])**2 for i in range(3))))
                        residuals.append(residual)
            errors.sort()
            cases.append({'pixel_jitter_px': jitter, 'common_extrinsic_shift_m': shift,
                          'fixtures': 18, 'accepted': len(errors), 'rejected': rejected,
                          'error_median_m': errors[len(errors)//2] if errors else None,
                          'error_p95_m': errors[math.ceil(0.95*len(errors))-1] if errors else None,
                          'error_max_m': max(errors) if errors else None,
                          'residual_max_m': max(residuals) if residuals else None})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); path = Path(args.output); path.mkdir(parents=True, exist_ok=False)
    report = {'feedback_level': 'offline_analytic_and_synthetic_rgb', 'official_platform': False,
              'official_score': None, 'runtime': code_identity(), 'analytic': analytic_scan(), 'rgb': []}
    for jitter in (0, 1, 2):
        for shift in (0.0, 0.01):
            for seed in (0, 1, 2):
                name = f'jitter-{jitter}-shift-{int(shift*1000)}mm-seed-{seed}'
                result = run_trial(path/name, seed=seed, noise=SensorNoise(jitter, shift))
                checked = replay(path/name)
                report['rgb'].append({'name': name, 'pixel_jitter_px': jitter, 'calibration_shift_m': shift,
                                      'seed': seed, 'result': result, 'replay': checked})
    (path/'robustness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'analytic_profiles': len(report['analytic']), 'rgb_trials': len(report['rgb']),
                      'rgb_successes': sum(row['result']['success'] for row in report['rgb']),
                      'report': str(path/'robustness.json')}, indent=2))


if __name__ == '__main__': main()

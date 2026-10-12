import argparse
from dataclasses import replace
import json
from .artifacts import load_json
from .contracts import Config
from .experiments import run_trial, batch, replay
from .preflight import environment_report, inspect_template
from .synthetic import SCENARIOS, SensorNoise


def main():
    parser = argparse.ArgumentParser(description='离线RGB闭环与记录校验；不是官方仿真')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run'); run.add_argument('--scenario', choices=SCENARIOS, default='nominal')
    run.add_argument('--seed', type=int, default=0); run.add_argument('--output', required=True)
    batches = sub.add_parser('batch'); batches.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    batches.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); batches.add_argument('--output', required=True)
    for cmd in (run, batches):
        cmd.add_argument('--config', help='strict JSON config overrides; saved in each trial')
        cmd.add_argument('--without-recovery', action='store_true')
        cmd.add_argument('--raw-observations', action='store_true', help='legacy single-frame baseline')
        cmd.add_argument('--pixel-jitter', type=int, default=0)
        cmd.add_argument('--calibration-shift-mm', type=float, default=0)
    replay_cmd = sub.add_parser('replay'); replay_cmd.add_argument('directory'); replay_cmd.add_argument('--expected-runtime-sha256')
    validate = sub.add_parser('validate-config'); validate.add_argument('config')
    preflight = sub.add_parser('preflight'); preflight.add_argument('--template')
    args = parser.parse_args()
    try:
        if args.command in ('run', 'batch'):
            config = Config.from_dict(load_json(args.config)) if args.config else Config(robust_observations=True)
            if args.raw_observations: config = replace(config, robust_observations=False)
            if args.without_recovery: config = replace(config, recovery_enabled=False)
            noise = SensorNoise(args.pixel_jitter, args.calibration_shift_mm/1000)
            if args.command == 'run': result = run_trial(args.output, args.scenario, args.seed, config, noise)
            else:
                result = batch(args.output, args.scenarios, args.seeds, config=config, noise=noise)
                result = {k: v for k, v in result.items() if k != 'trials'}
        elif args.command == 'replay': result = replay(args.directory, args.expected_runtime_sha256)
        elif args.command == 'validate-config':
            from dataclasses import asdict
            result = asdict(Config.from_dict(load_json(args.config)))
        else: result = inspect_template(args.template) if args.template else environment_report()
    except (ValueError, OSError) as error: parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()

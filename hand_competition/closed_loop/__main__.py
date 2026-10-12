import argparse
import json
from .contracts import Config
from .experiments import run_trial, batch, replay
from .preflight import environment_report, inspect_template
from .synthetic import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description='RGB闭环实验；当前仅合成运动学场景，不是官方仿真')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run'); run.add_argument('--scenario', choices=SCENARIOS, default='nominal')
    run.add_argument('--seed', type=int, default=0); run.add_argument('--output', required=True)
    run.add_argument('--without-recovery', action='store_true')
    batches = sub.add_parser('batch'); batches.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    batches.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); batches.add_argument('--output', required=True)
    batches.add_argument('--without-recovery', action='store_true')
    replay_cmd = sub.add_parser('replay'); replay_cmd.add_argument('directory')
    preflight = sub.add_parser('preflight'); preflight.add_argument('--template')
    args = parser.parse_args()
    try:
        if args.command == 'run':
            result = run_trial(args.output, args.scenario, args.seed, Config(recovery_enabled=not args.without_recovery))
        elif args.command == 'batch':
            result = batch(args.output, args.scenarios, args.seeds, not args.without_recovery)
            # Full per-trial evidence stays in summary.json, not noisy stdout.
            result = {k: v for k, v in result.items() if k != 'trials'}
        elif args.command == 'replay': result = replay(args.directory)
        else: result = inspect_template(args.template) if args.template else environment_report()
    except (ValueError, OSError) as error: parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()

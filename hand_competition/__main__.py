"""Run only the offline annotation calculator."""

import argparse
import json

from .evaluation import append_review, evaluate, load_record


def main():
    parser = argparse.ArgumentParser(description="非官方离线最终状态记分；不连接机器人")
    parser.add_argument("record", help="JSON annotation file; not robot commands")
    parser.add_argument("--log", help="optional append-only local JSONL path (single writer)")
    args = parser.parse_args()
    try:
        record = load_record(args.record)
        result = append_review(args.log, record) if args.log else evaluate(record)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

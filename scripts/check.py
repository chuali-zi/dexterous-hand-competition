#!/usr/bin/env python3
"""Local CI entrypoint: tests + repository docs/data integrity, no network."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): return 1
    count = 0
    for path in ROOT.rglob('*.md'):
        if any(part in ('.git', 'runs', '.venv') for part in path.parts): continue
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#', 'mailto:')): continue
            if not (path.parent/target.split('#')[0]).exists():
                print(f'Broken relative link: {path.relative_to(ROOT)} -> {target}', file=sys.stderr)
                return 1
            count += 1
    for directory in ('configs', 'examples', 'docs/evidence'):
        for path in (ROOT/directory).glob('*.json'): json.loads(path.read_text(encoding='utf-8'))
    lock = json.loads((ROOT/'docs/evidence/source-lock.json').read_text())
    for repo in lock['repositories']:
        if not re.fullmatch(r'[0-9a-f]{40}', repo['commit']): raise ValueError('source not pinned')
        for item in repo['files']:
            if item.get('status') == 200:
                if repo['commit'] not in item['url'] or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                    raise ValueError('invalid source lock entry')
    print(f'PASS: {result.testsRun} tests; {count} relative Markdown links; JSON and source-lock structure')
    return 0


if __name__ == '__main__': raise SystemExit(main())

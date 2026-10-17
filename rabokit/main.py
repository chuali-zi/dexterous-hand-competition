"""Copy this file to /workspace/agent_system/main.py for the platform Run button."""
from pathlib import Path
import runpy
import sys

project = Path(__file__).resolve().parent / 'competition'
entry = project / 'scripts' / 'rabokit_probe.py'
if not entry.is_file():
    raise SystemExit('Clone the GitHub project into /workspace/agent_system/competition first')
sys.path.insert(0, str(project))
runpy.run_path(str(entry), run_name='__main__')

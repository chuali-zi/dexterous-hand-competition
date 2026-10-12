"""Read-only local discovery. Never initializes SDK, opens an app or logs in."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import platform
import shutil


def environment_report():
    modules = {}
    for name in ('rabo_robolib', 'rabo_robocap', 'rclpy', 'cv2', 'playwright'):
        spec = importlib.util.find_spec(name)
        modules[name] = {'available': spec is not None,
                         'origin': spec.origin if spec else None}
    return {'python': platform.python_version(), 'modules': modules,
            'executables': {name: shutil.which(name) for name in ('chromium', 'ros2', 'docker')},
            'official_platform_verified': False, 'sdk_initialized': False,
            'note': 'No credentials/cookies inspected. Availability does not establish permission or free execution.'}


def inspect_template(directory):
    """AST/file inspection only. No downloaded source is imported or executed.

    Scan Python and dependency declarations, exclude hidden/secret directories.
    Findings are review hints rather than a complete semantic leakage audit.
    """
    root = Path(directory).resolve()
    if not root.is_dir(): raise ValueError('template directory does not exist')
    findings = []; files = {}; imports = set(); parse_errors = []
    ignored = {'.git', '.venv', 'node_modules', '__pycache__', 'data', 'runs', 'checkpoints'}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(p in ignored or p.startswith('.') for p in relative.parts): continue
        if path.is_symlink() or not path.is_file(): continue
        if path.suffix != '.py' and path.name not in ('requirements.txt', 'pyproject.toml', 'package.xml'): continue
        data = path.read_bytes(); files[str(relative)] = hashlib.sha256(data).hexdigest()
        if path.suffix != '.py': continue
        try: tree = ast.parse(data)
        except (SyntaxError, UnicodeDecodeError):
            parse_errors.append(str(relative)); continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module)
            elif isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                if name in ('SetEntityPose', 'set_entity_pose', 'attach', 'detach', 'lookup_transform', 'get_entity_pose', 'get_object_pose'):
                    findings.append({'file': str(relative), 'line': node.lineno,
                                     'kind': 'control_input_or_scene_manipulation_review', 'symbol': name})
                elif name == 'move_to' and any(k.arg == 'linear' for k in node.keywords):
                    findings.append({'file': str(relative), 'line': node.lineno,
                                     'kind': 'joint_interpolation_not_cartesian_guarantee', 'symbol': name})
    return {'template_path': str(root), 'files_sha256': files, 'imports': sorted(imports),
            'findings': findings, 'parse_errors': parse_errors,
            'executed_source': False, 'official_origin_verified': False,
            'note': 'Human/parent must identify actual official origin/version; AST hints do not certify safety.'}

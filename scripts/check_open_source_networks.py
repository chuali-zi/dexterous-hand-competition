"""Load upstream weights on CPU. Synthetic inputs test neural execution only.

Clone GraspXL and DexGrasp-Zero-code into --repos and fetch their small model
folders first. This does not validate sensor adapters, actions, or grasp success.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def load_module(path):
    spec = importlib.util.spec_from_file_location('upstream_network', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(repos):
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(75)
    folder = repos / 'GraspXL'
    module = load_module(folder / 'raisimGymTorch/raisimGymTorch/algo/ppo/module.py')
    weights = torch.load(folder / 'raisimGymTorch/data_all/allegro_floating/full_18000_r.pt',
                         weights_only=True, map_location='cpu')
    model = module.MLP([128, 128], torch.nn.LeakyReLU, 210, 22)
    model.load_state_dict(weights['actor_architecture_state_dict'], strict=True)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.architecture(torch.zeros(32, 210))
    results = {'GraspXL': {'shape': list(output.shape), 'finite': bool(output.isfinite().all()),
                          'seconds': time.perf_counter() - start, 'strict_weights': True,
                          'scope': 'synthetic normalized observations; not a physics rollout'}}
    folder = repos / 'DexGrasp-Zero-code'
    module = load_module(folder / 'algo/ppo/module.py')
    config = yaml.safe_load((folder / 'model/multi_hand/model_gnn_cfg_L10_v4.yaml').read_text(encoding='utf-8'))

    class SyntheticGraph:
        """Exercises learned layers; intentionally does not pretend to map a hand."""
        infoB_input_dim = 33

        def vec_to_Graph(self, obs, handtype):
            batch, nodes = obs.shape[0], 15
            return {'nodes': torch.zeros(batch, nodes, 16),
                    'edges': torch.eye(nodes).expand(batch, nodes, nodes),
                    'global': torch.zeros(batch, 24), 'infoB_nodes': torch.zeros(batch, nodes, 33),
                    'infoC_nodes': torch.ones(batch, nodes, 3)}

        def G_to_action(self, rt, joints, handtype):
            return torch.cat([rt, joints.flatten(1)], 1)  # raw heads, not robot actions

    model = module.GNN.from_config(config, SyntheticGraph(), {0: {'obs_dim': 1}})
    weights = torch.load(folder / 'model/multi_hand/full_2700_r.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(weights['actor_architecture_state_dict'], strict=True)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(torch.zeros(32, 1), 0)
    results['DexGrasp-Zero'] = {'raw_head_shape': list(output.shape), 'finite': bool(output.isfinite().all()),
                               'seconds': time.perf_counter() - start, 'strict_weights': True,
                               'scope': 'synthetic graph core; no URDF observations or action mapping'}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repos', type=Path, default=Path('runs/open-source-75/repos'))
    parser.add_argument('--deps', type=Path, default=Path('runs/open-source-75/deps'))
    parser.add_argument('--out', type=Path, default=Path('runs/open-source-75/neural-smoke.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.deps.resolve()))
    results = run(args.repos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(result['finite'] for result in results.values()) else 1)

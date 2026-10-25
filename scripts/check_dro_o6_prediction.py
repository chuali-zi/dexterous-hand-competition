"""CPU inference and geometric reconstruction of O6/cube inputs with DRO-Grasp.

O6 is not a supported pretrained hand. This is an out-of-distribution input
experiment, not the upstream benchmark and not a dynamic grasp test.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def sample_inputs(model_path, count=512):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(75)
    cloud = []
    for geom in range(model.ngeom):
        if not model.body(model.geom_bodyid[geom]).name.startswith('rh_'):
            continue
        mesh = model.geom_dataid[geom]
        if mesh < 0:
            continue
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        start = model.mesh_faceadr[mesh]
        faces = model.mesh_face[start:start + model.mesh_facenum[mesh]]
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        chosen = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
        uv = rng.random((count, 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        points = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
        cloud.append(points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    cloud = np.concatenate(cloud)
    # Deterministic farthest-point sampling, matching the upstream sampling idea.
    indices = [0]
    distances = np.full(len(cloud), np.inf)
    for _ in range(count - 1):
        distances = np.minimum(distances, np.linalg.norm(cloud - cloud[indices[-1]], axis=1))
        indices.append(int(np.argmax(distances)))
    hand = cloud[indices]
    hand -= hand.mean(axis=0)
    cube = rng.uniform(-.025, .025, (count, 3))
    axis = rng.integers(0, 3, count)
    cube[np.arange(count), axis] = rng.choice([-.025, .025], count)
    return hand.astype(np.float32), cube.astype(np.float32)


def run(repo, checkpoint, model_path):
    import torch
    sys.path.insert(0, str(repo.resolve()))
    from model.network import Network
    from utils.multilateration import multilateration
    torch.set_num_threads(2)
    torch.manual_seed(75)
    model = Network(SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None,
                                    center_pc=True, block_computing=True), 'validate')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    hand, cube = sample_inputs(model_path)
    with torch.no_grad():
        known = torch.cdist(torch.from_numpy(hand[None]), torch.from_numpy(cube[None]))
        restored = multilateration(known, torch.from_numpy(cube[None]))
        control = float((restored - torch.from_numpy(hand[None])).square().mean().sqrt())
    rows = []
    predictions = []
    for seed in [75, 76, 77]:
        torch.manual_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            distances = model(torch.from_numpy(hand[None]), torch.from_numpy(cube[None]))['dro']
            points = multilateration(distances, torch.from_numpy(cube[None]))
            reconstructed = torch.cdist(points, torch.from_numpy(cube[None]))
            residual = float((reconstructed - distances).square().mean().sqrt())
        rows.append({'seed': seed, 'seconds': time.perf_counter() - start,
                     'finite': bool(distances.isfinite().all() and points.isfinite().all()),
                     'distance_shape': list(distances.shape), 'distance_reconstruction_rmse_m': residual,
                     'predicted_hand_extent_m': (points.max(1).values - points.min(1).values)[0].tolist()})
        predictions.append(points[0].numpy())
    return {'strict_weights': True, 'analytic_control_position_rmse_m': control, 'results': rows,
            'scope': 'O6 unseen-hand inference and point reconstruction; no joint IK, collision, lift or hold validation'}, hand, cube, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('runs/open-source-75/repos/DRO-Grasp'))
    parser.add_argument('--model', type=Path, default=Path('runs/open-source-75/o6-physics/demo_50g/model.xml'))
    parser.add_argument('--out', type=Path, default=Path('runs/open-source-75/dro-prediction'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path('runs/mujoco-runtime').resolve()))
    result, hand, cube, predictions = run(args.repo, args.checkpoint, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez(args.out / 'points.npz', hand=hand, cube=cube, predictions=predictions)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(r['finite'] for r in result['results']) else 1)

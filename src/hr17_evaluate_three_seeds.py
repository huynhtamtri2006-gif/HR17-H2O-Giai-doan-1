"""Matched deterministic evaluation: 3 training seeds x initial/final, no training."""
DIAGNOSTIC_SOURCE = '"""Pre-reset diagnostics derived from the successful Colab smoke test."""\nimport json\nfrom pathlib import Path\n\nimport isaacgym\nfrom isaacgym import gymapi, gymutil\nimport torch\nfrom torch import nn\nfrom easydict import EasyDict\n\nfrom legged_gym.envs import LeggedRobot\nfrom legged_gym.utils.helpers import set_seed\n\n\nimport csv\nimport argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--checkpoint\', choices=[\'model_initial.pt\', \'model_200.pt\'], required=True)\nparser.add_argument(\'--mass-scale\', type=float, choices=[1.0], default=1.0)\nparser.add_argument(\'--output\', type=Path, required=True)\nparser.add_argument(\'--max-steps\', type=int, default=300)\nparser.add_argument(\'--run-dir\', type=Path, default=Path(\'/content/hr17_training/H2O_seed1_20261116_005911\'))\nparser.set_defaults(controller=\'checkpoint\')\ncli_args = parser.parse_args()\nimport hashlib\nimport numpy as np\nfrom collections import Counter\nfrom phc.utils import torch_utils as phc_torch_utils\n\n\nclass DiagnosticRobot(LeggedRobot):\n    """Observe before reset_idx; use the repository\'s original termination logic."""\n\n    def __init__(self, *args, **kwargs):\n        self.diag_enabled = False\n        self.diag_rows = []\n        self.diag_pred = []\n        self.diag_ref = []\n        self.diag_episode = 0\n        super().__init__(*args, **kwargs)\n        self.diag_enabled = True\n\n    def _process_rigid_body_props(self, props, env_id):\n        props = super()._process_rigid_body_props(props, env_id)\n        assert env_id == 0\n        self.mass_before_scaling = [float(p.mass) for p in props]\n        for prop in props:\n            prop.mass *= cli_args.mass_scale\n        self.mass_requested = [float(p.mass) for p in props]\n        return props\n\n    def check_termination(self):\n        super().check_termination()\n        if not self.diag_enabled:\n            return\n        assert self.num_envs == 1\n\n        offset = self.env_origins + self.env_origins_init_3Doffset\n        motion_time = self.episode_length_buf * self.dt + self.motion_start_times\n        reference = self._get_state_from_motionlib_cache_trimesh(\n            self.motion_ids, motion_time, offset=offset\n        )\n        positions = self._rigid_body_pos\n        rotations = self._rigid_body_rot\n        extended = phc_torch_utils.my_quat_rotate(\n            rotations[:, self.extend_body_parent_ids].reshape(-1, 4),\n            self.extend_body_pos.reshape(-1, 3),\n        ).reshape(1, -1, 3) + positions[:, self.extend_body_parent_ids]\n        predicted = torch.cat((positions, extended), dim=1)\n        target = reference[\'rg_pos_t\']\n        assert predicted.shape == target.shape, (predicted.shape, target.shape)\n        assert torch.isfinite(predicted).all() and torch.isfinite(target).all()\n\n        global_error = (predicted - target).norm(dim=-1).mean().item() * 1000\n        local_error = ((predicted - predicted[:, [0]])\n                       - (target - target[:, [0]])).norm(dim=-1).mean().item() * 1000\n        body_distance = (positions - reference[\'rg_pos\']).norm(dim=-1).mean().item()\n\n        contacts = self.contact_forces[0, self.termination_contact_indices].norm(dim=-1)\n        hit_indices = self.termination_contact_indices[contacts > 1.0].tolist()\n        hit_names = [self._body_list[int(i)] for i in hit_indices]\n        reasons = []\n        asset = self.cfg.asset\n        scales = asset.termination_scales\n        if hit_names:\n            reasons.append(\'body_contact\')\n        if asset.terminate_by_gravity:\n            if abs(self.projected_gravity[0, 0].item()) > scales.gravity_x:\n                reasons.append(\'gravity_x\')\n            if abs(self.projected_gravity[0, 1].item()) > scales.gravity_y:\n                reasons.append(\'gravity_y\')\n        if asset.terminate_by_low_height and self.root_states[0, 2].item() < scales.base_height:\n            reasons.append(\'low_height\')\n        if asset.terminate_by_ref_motion_distance and self._recovery_counter[0].item() <= 0:\n            if body_distance > scales.max_ref_motion_distance:\n                reasons.append(\'reference_distance\')\n        if bool(self.time_out_buf[0].item()):\n            reasons.append(\'motion_timeout\')\n        done = bool(self.reset_buf[0].item())\n        if done and not reasons:\n            reasons.append(\'other_termination\')\n\n        self.diag_rows.append(dict(\n            frame=len(self.diag_rows) + 1,\n            episode=self.diag_episode,\n            episode_step=int(self.episode_length_buf[0].item()),\n            motion_time_s=float(motion_time[0].item()),\n            global_mpjpe_mm=global_error,\n            root_relative_mpjpe_mm=local_error,\n            mean_body_distance_m=body_distance,\n            root_height_m=float(self.root_states[0, 2].item()),\n            gravity_x=float(self.projected_gravity[0, 0].item()),\n            gravity_y=float(self.projected_gravity[0, 1].item()),\n            done=done,\n            reasons=\';\'.join(reasons) if done else \'\',\n            contact_bodies=\';\'.join(hit_names) if done else \'\',\n        ))\n        self.diag_pred.append(predicted[0].detach().cpu().numpy().copy())\n        self.diag_ref.append(target[0].detach().cpu().numpy().copy())\n        self.diag_rows[-1].update(\n            requested_action_abs_max=self.diag_action_abs_max,\n            action_clip_fraction=self.diag_action_clip_fraction,\n            joint_target_error_rad=float((self.diag_joint_target - self.dof_pos).abs().mean().item()),\n            torque_abs_max=float(self.torques.abs().max().item()),\n            torque_limit_fraction=float((self.torques.abs() >= 0.999 * self.torque_limits).float().mean().item()),\n        )\n        if done:\n            self.diag_episode += 1\n\n\nrepo = Path("/content/hr17_repo")\nfolder = cli_args.run_dir\nmetadata = json.loads((folder / \'run_metadata.json\').read_text())\nassert metadata[\'status\'] == \'completed\'\nassert metadata[\'seed\'] in [1, 2, 3] and metadata[\'iterations\'] == 200\nassert hashlib.sha256((folder / \'model_200.pt\').read_bytes()).hexdigest() == metadata[\'final_checkpoint_sha256\']\noutput = cli_args.output\noutput.mkdir(parents=True, exist_ok=False)\nimport shutil\nshutil.copy2(__file__, output / Path(__file__).name)\nfor filename in (\'run_metadata.json\', \'env_cfg.json\', \'train_cfg.json\'):\n    shutil.copy2(folder / filename, output / (\'source_\' + filename))\n\n# Đọc lại cấu hình gốc.\ncfg = EasyDict(json.loads((folder / "env_cfg.json").read_text()))\ntrain_cfg = json.loads((folder / "train_cfg.json").read_text())\n\n# Chạy một môi trường, không mở cửa sổ.\ncfg.seed = 1\ncfg.train.seed = 1\ncfg.num_envs = 1\ncfg.env.num_envs = 1\ncfg.headless = True\ncfg.sim_device = "cuda:0"\ncfg.rl_device = "cuda:0"\ncfg.test = True\ncfg.env.test = True\ncfg.env.im_eval = False\ncfg.train.distill = False\ncfg.use_wandb = False\ncfg.viewer.debug_viz = False\ncfg.motion.visualize = False\ncfg.motion.realtime_vr_keypoints = False\ncfg.rewards.penalty_scale = 1.0\n\n# Đường dẫn tới mô hình và motion.\nresources = repo / "legged_gym" / "resources"\ncfg.asset.file = str(resources / "robots/h1/urdf/h1.urdf")\ncfg.motion.motion_file = str(resources / "motions/h1/stable_punch.pkl")\ncfg.motion.skeleton_file = str(resources / "robots/h1/xml/h1.xml")\ncfg.motion.marker_file = str(\n    resources / "objects/Marker/traj_marker.urdf"\n)\n\nassert Path(cfg.asset.file).exists(), f"Thiếu file: {cfg.asset.file}"\nfor name in ("motion_file", "skeleton_file"):\n    assert Path(cfg.motion[name]).exists(), (\n        f"Thiếu file: {cfg.motion[name]}"\n    )\n\n# Địa hình phẳng dạng trimesh nhỏ.\ncfg.terrain.curriculum = False\ncfg.terrain.mesh_type = "trimesh"\ncfg.terrain.num_rows = 1\ncfg.terrain.num_cols = 1\ncfg.terrain.max_init_terrain_level = 0\ncfg.terrain.border_size = 5\ncfg.terrain.terrain_types = ["flat"]\ncfg.terrain.terrain_proportions = [1.0]\n\n# Chỉ tắt các cờ ngẫu nhiên kiểu bool.\ncfg.noise.add_noise = False\nfor key, value in list(cfg.domain_rand.items()):\n    if key.startswith("randomize_") and isinstance(value, bool):\n        cfg.domain_rand[key] = False\n\nassert isinstance(\n    cfg.domain_rand.randomize_link_body_names, list\n), "Danh sách khâu robot phải được giữ nguyên"\n\ncfg.domain_rand.push_robots = False\ncfg.domain_rand.motion_package_loss = False\ncfg.domain_rand.born_offset = False\ncfg.domain_rand.born_heading_randomization = False\ncfg.motion.resample_motions_for_envs = False\n\n(output / "config_requested.json").write_text(\n    json.dumps(cfg, indent=2)\n)\n\nset_seed(cfg.seed)\n\nsim_params = gymapi.SimParams()\ngymutil.parse_sim_config(cfg.sim, sim_params)\nsim_params.use_gpu_pipeline = True\nsim_params.physx.use_gpu = True\n\n# Khôi phục actor, không nạp optimizer.\ncheckpoint = torch.load(\n    folder / cli_args.checkpoint,\n    map_location="cpu",\n    weights_only=True,\n)\n\ndims = (\n    [cfg.env.num_observations]\n    + train_cfg["policy"]["actor_hidden_dims"]\n    + [cfg.env.num_actions]\n)\nassert train_cfg["policy"]["activation"].lower() == "elu"\n\nlayers = []\nfor i in range(len(dims) - 1):\n    layers.append(nn.Linear(dims[i], dims[i + 1]))\n    if i < len(dims) - 2:\n        layers.append(nn.ELU())\n\nactor = nn.Sequential(*layers)\nactor.load_state_dict({\n    key[len("actor."):]: value\n    for key, value in checkpoint["model_state_dict"].items()\n    if key.startswith("actor.")\n}, strict=True)\nactor = actor.to("cuda:0").eval()\n\nprint("ĐANG KHỞI TẠO MÔI TRƯỜNG H2O...")\nenv = DiagnosticRobot(\n    cfg=cfg,\n    sim_params=sim_params,\n    physics_engine=gymapi.SIM_PHYSX,\n    sim_device="cuda:0",\n    headless=True,\n)\n\ntry:\n    # Read the actual actor properties back from Isaac Gym before evaluating.\n    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])\n    actual = np.asarray([float(p.mass) for p in props])\n    original = np.asarray(env.mass_before_scaling)\n    requested = np.asarray(env.mass_requested)\n    assert (original > 0).any() and np.isfinite(actual).all()\n    assert np.allclose(requested, original * cli_args.mass_scale, rtol=1e-5, atol=1e-5)\n    assert np.allclose(actual, requested, rtol=1e-4, atol=1e-5), (actual, requested)\n    shapes = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])\n    audit = dict(\n        mass_scale=cli_args.mass_scale,\n        body_names=env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]),\n        original_mass_kg=original.tolist(), requested_mass_kg=requested.tolist(),\n        actual_mass_kg=actual.tolist(), actual_total_mass_kg=float(actual.sum()),\n        original_total_mass_kg=float(original.sum()),\n        mass_readback_verified=True,\n        inertia_recomputed_by_repository=True,\n        actor_shape_friction=[float(p.friction) for p in shapes],\n        terrain_static_friction=float(cfg.terrain.static_friction),\n        terrain_dynamic_friction=float(cfg.terrain.dynamic_friction),\n        control_dt_s=float(env.dt), seed=int(cfg.seed),\n    )\n    (output / \'physics_audit.json\').write_text(json.dumps(audit, indent=2))\n    print(\'PHYSICS VERIFIED:\', cli_args.mass_scale, \'| total mass kg:\', actual.sum())\n    obs = env.get_observations()\n    np.savez_compressed(output / \'initial_state.npz\',\n                        root=env.root_states.detach().cpu().numpy(),\n                        dof=env.dof_pos.detach().cpu().numpy(),\n                        observation=obs.detach().cpu().numpy())\n    actor_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}\n    assert obs.shape == (1, 138)\n    print(\'CHECKPOINT:\', cli_args.checkpoint, \'| first episode only; up to\', cli_args.max_steps, \'steps\')\n    with torch.inference_mode():\n        for step in range(cli_args.max_steps):\n            assert torch.isfinite(obs).all(), \'Observation contains NaN/Inf\'\n            if cli_args.controller == \'checkpoint\':\n                actions = actor(obs)\n            elif cli_args.controller == \'zero\':\n                # Zero action means PD holds default joint targets, not zero torque.\n                actions = torch.zeros_like(env.dof_pos)\n            else:\n                assert cfg.control.control_type == \'P\'\n                assert cfg.control.action_scale > 0\n                offset = env.env_origins + env.env_origins_init_3Doffset\n                next_time = (env.episode_length_buf + 1) * env.dt + env.motion_start_times\n                reference = env._get_state_from_motionlib_cache_trimesh(\n                    env.motion_ids, next_time, offset=offset\n                )\n                actions = (reference[\'dof_pos\'] - env.default_dof_pos) / cfg.control.action_scale\n            clipped = torch.clamp(actions, -cfg.normalization.clip_actions,\n                                  cfg.normalization.clip_actions)\n            env.diag_joint_target = (clipped * cfg.control.action_scale\n                                     + env.default_dof_pos).detach().clone()\n            env.diag_action_abs_max = float(actions.abs().max().item())\n            env.diag_action_clip_fraction = float((actions != clipped).float().mean().item())\n            assert torch.isfinite(actions).all(), \'Action contains NaN/Inf\'\n            previous_count = len(env.diag_rows)\n            obs, _, rewards, dones, infos = env.step(actions)\n            assert len(env.diag_rows) == previous_count + 1\n            assert torch.isfinite(rewards).all(), \'Reward contains NaN/Inf\'\n            row = env.diag_rows[-1]\n            row[\'reward\'] = float(rewards[0].item())\n            if row[\'done\']:\n                print(f"Reset at frame {row[\'frame\']}, episode step {row[\'episode_step\']}: "\n                      f"{row[\'reasons\']} | contacts={row[\'contact_bodies\']} | "\n                      f"global error={row[\'global_mpjpe_mm\']:.2f} mm")\n                break\n\n    rows = env.diag_rows\n    assert 1 <= len(rows) <= cli_args.max_steps\n    assert all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())\n    with (output / \'frames.csv\').open(\'w\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))\n        writer.writeheader()\n        writer.writerows(rows)\n    np.savez_compressed(output / \'positions_before_reset.npz\',\n                        robot=np.stack(env.diag_pred), reference=np.stack(env.diag_ref))\n    reason_counts = Counter(reason for row in rows if row[\'done\']\n                            for reason in row[\'reasons\'].split(\';\') if reason)\n    summary = dict(\n        purpose=\'matched deterministic initial/final comparison for three training seeds; one motion, not paper evaluation\',\n        training_seed=int(metadata[\'seed\']),\n        evaluation_seed=1,\n        mass_scale=cli_args.mass_scale,\n        actor_weights_unchanged=True,\n        maximum_control_steps=cli_args.max_steps,\n        controller=cli_args.controller,\n        checkpoint_name=cli_args.checkpoint,\n        checkpoint_used_for_actions=(cli_args.controller == \'checkpoint\'),\n        max_requested_action=float(max(row[\'requested_action_abs_max\'] for row in rows)),\n        mean_action_clip_fraction=float(np.mean([row[\'action_clip_fraction\'] for row in rows])),\n        mean_torque_limit_fraction=float(np.mean([row[\'torque_limit_fraction\'] for row in rows])),\n        metric_scope=\'first episode only, including terminal frame; compare MPJPE on common nonterminal prefix\',\n        checkpoint_provenance=\'user fresh PPO run; archived hyperparameter source provenance unknown\',\n        training_run=folder.name,\n        training_initialization=metadata[\'initialization\'],\n        deterministic_actor=True,\n        checkpoint_sha256=hashlib.sha256((folder / cli_args.checkpoint).read_bytes()).hexdigest(),\n        seed=cfg.seed,\n        control_dt_s=float(env.dt),\n        frame_count=len(rows),\n        reset_count=sum(row[\'done\'] for row in rows),\n        first_reset_step=next((r[\'episode_step\'] for r in rows if r[\'done\']), None),\n        first_reset_time_s=next((r[\'motion_time_s\'] for r in rows if r[\'done\']), None),\n        reset_reason_counts=dict(reason_counts),\n        observed_frame_global_mpjpe_mm=float(np.mean([r[\'global_mpjpe_mm\'] for r in rows])),\n        observed_frame_root_relative_mpjpe_mm=float(np.mean([r[\'root_relative_mpjpe_mm\'] for r in rows])),\n        num_body_points=int(env.diag_pred[0].shape[0]),\n        reference_distance_termination_threshold_m=float(cfg.asset.termination_scales.max_ref_motion_distance),\n        success_rate=None,\n        note=\'Multiple causes may occur in one reset; success rate is not measured here.\',\n    )\n    (output / \'summary.json\').write_text(json.dumps(summary, indent=2))\n    print(\'\\nDIAGNOSTIC SUMMARY:\')\n    print(json.dumps(summary, indent=2))\n    print(\'CHAN DOAN RESET HOAN TAT\')\n    print(\'Results:\', output)\nfinally:\n    env.gym.destroy_sim(env.sim)\n'
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile

CASES = [(seed, stage, filename) for seed in [1, 2, 3]
         for stage, filename in [('initial', 'model_initial.pt'), ('final', 'model_200.pt')]]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def capture(command, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout

def lines(text):
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')}

def summarize(output, source_hashes):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {}
    initial = []
    configs = []
    for seed, stage, filename in CASES:
        label = 'seed%d_%s' % (seed, stage)
        folder = output / label
        summary = load(folder / 'summary.json')
        physics = load(folder / 'physics_audit.json')
        assert summary['training_seed'] == seed and summary['evaluation_seed'] == summary['seed'] == 1
        assert summary['checkpoint_sha256'] == source_hashes[label]
        assert summary['actor_weights_unchanged']
        assert physics['mass_scale'] == 1.0 and physics['mass_readback_verified']
        config = load(folder / 'config_requested.json')
        assert config['seed'] == config['train']['seed'] == 1
        configs.append(config)
        with np.load(folder / 'initial_state.npz') as state:
            initial.append({key: state[key].copy() for key in state.files})
        with (folder / 'frames.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == summary['frame_count'] and rows
        assert [int(row['episode_step']) for row in rows] == list(range(1, len(rows) + 1))
        assert all(int(row['episode']) == 0 for row in rows)
        assert all(row['done'] == 'False' for row in rows[:-1])
        with np.load(folder / 'positions_before_reset.npz') as positions:
            robot, reference = positions['robot'], positions['reference']
            assert robot.shape == reference.shape == (len(rows), 22, 3)
            assert np.isfinite(robot).all() and np.isfinite(reference).all()
            g = np.linalg.norm(robot - reference, axis=-1).mean(axis=-1) * 1000
            l = np.linalg.norm((robot - robot[:, [0]]) - (reference - reference[:, [0]]), axis=-1).mean(axis=-1) * 1000
            for key, calculated in [('global_mpjpe_mm', g), ('root_relative_mpjpe_mm', l)]:
                assert np.allclose(calculated, [float(r[key]) for r in rows], rtol=1e-5, atol=1e-3)
            data[label] = dict(rows=rows, summary=summary, physics=physics, reference=reference.copy())
    assert all(config == configs[0] for config in configs), 'Evaluation configs differ'
    base_physics = data['seed1_initial']['physics']
    for label, entry in data.items():
        for key in ['body_names', 'actual_mass_kg', 'actor_shape_friction', 'terrain_static_friction', 'terrain_dynamic_friction', 'control_dt_s']:
            assert entry['physics'][key] == base_physics[key], (label, key)
    initial_checks = []
    for (seed, stage, _), state in zip(CASES[1:], initial[1:]):
        for key in ['root', 'dof', 'observation']:
            assert state[key].shape == initial[0][key].shape
            assert np.isfinite(state[key]).all() and np.isfinite(initial[0][key]).all()
            delta = np.abs(state[key].astype(np.float64) - initial[0][key].astype(np.float64))
            tolerance = np.full(delta.shape, 1e-6)
            if key == 'observation': tolerance[:, 41:44] = 2e-6
            passed = bool(np.all(delta <= tolerance))
            initial_checks.append(dict(case='seed%d_%s' % (seed, stage), field=key,
                                       max_absolute_difference=float(delta.max()), within_tolerance=passed))
    write(output / 'initial_condition_audit.json', dict(
        checks=initial_checks, default_absolute_tolerance=1e-6,
        angular_velocity_observation_tolerance=2e-6, relative_tolerance=0,
        angular_velocity_observation_indices=[41, 42, 43],
        note='Tolerances fixed before evaluation, based on prior numerical audit.'))
    assert all(check['within_tolerance'] for check in initial_checks), 'Initial states differ'
    horizon = min(next((i for i, r in enumerate(d['rows']) if r['done'] == 'True'), len(d['rows'])) for d in data.values())
    assert horizon > 0, 'No common nonterminal prefix'
    baseline_ref = data['seed1_initial']['reference'][:horizon]
    results = []
    for seed, stage, filename in CASES:
        label = 'seed%d_%s' % (seed, stage)
        entry = data[label]
        rows = entry['rows']
        prefix = rows[:horizon]
        assert all(r['done'] == 'False' for r in prefix)
        assert np.allclose(entry['reference'][:horizon], baseline_ref, rtol=0, atol=1e-6)
        assert np.allclose([float(r['motion_time_s']) for r in prefix],
                           [float(r['motion_time_s']) for r in data['seed1_initial']['rows'][:horizon]], rtol=0, atol=1e-6)
        dt = entry['summary']['control_dt_s']
        ended = rows[-1]['done'] == 'True'
        results.append(dict(training_seed=seed, stage=stage, checkpoint=filename,
            evaluation_seed=1, common_nonterminal_steps=horizon, common_time_s=horizon * dt,
            global_mpjpe_mm=statistics.mean(float(r['global_mpjpe_mm']) for r in prefix),
            root_relative_mpjpe_mm=statistics.mean(float(r['root_relative_mpjpe_mm']) for r in prefix),
            first_reset_time_s=(len(rows) * dt if ended else None),
            observed_duration_s=len(rows) * dt, frames=len(rows),
            end_reason=(rows[-1]['reasons'] if ended else 'step_limit_censored'),
            checkpoint_sha256=source_hashes[label]))
    aggregate = {}
    for stage in ['initial', 'final']:
        selected = [r for r in results if r['stage'] == stage]
        aggregate[stage] = {}
        for key in ['global_mpjpe_mm', 'root_relative_mpjpe_mm', 'first_reset_time_s']:
            values = [r[key] for r in selected if r[key] is not None]
            aggregate[stage][key] = dict(mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
                n=len(values), censoring_present=len(values) < 3)
    paired = []
    for seed in [1, 2, 3]:
        before = next(r for r in results if r['training_seed'] == seed and r['stage'] == 'initial')
        after = next(r for r in results if r['training_seed'] == seed and r['stage'] == 'final')
        paired.append(dict(training_seed=seed,
            global_change_mm=after['global_mpjpe_mm'] - before['global_mpjpe_mm'],
            root_relative_change_mm=after['root_relative_mpjpe_mm'] - before['root_relative_mpjpe_mm'],
            reset_time_change_s=(after['first_reset_time_s'] - before['first_reset_time_s']
                                 if before['first_reset_time_s'] is not None and after['first_reset_time_s'] is not None else None)))
    report = dict(scope='six matched deterministic first-episode rollouts; one motion, eval seed 1; not paper evaluation',
                  training_seeds=[1, 2, 3], evaluation_seed=1, common_nonterminal_steps=horizon,
                  initial_conditions_within_documented_tolerance=True, configs_equal=True,
                  physical_parameters_equal=True, checkpoint_weights_unchanged=True,
                  metrics_recomputed_from_npz=True, success_rate=None,
                  results=results, aggregate=aggregate, paired_changes=paired,
                  note='Aggregate spread is across training seeds (n=3), not repeated evaluation trials. '
                       'MPJPE excludes all terminal frames. Timeout is not automatically classified as success.')
    write(output / 'evaluation_comparison.json', report)
    with (output / 'evaluation_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ['global_mpjpe_mm', 'root_relative_mpjpe_mm', 'first_reset_time_s'],
                             ['Global MPJPE (mm)', 'Root-relative MPJPE (mm)', 'First reset elapsed time (s)']):
        for offset, stage in [(-.18, 'initial'), (.18, 'final')]:
            values = [next(r[key] for r in results if r['training_seed'] == seed and r['stage'] == stage) for seed in [1, 2, 3]]
            ax.bar([seed + offset for seed in [1, 2, 3]], [float('nan') if v is None else v for v in values], width=.36, label=stage)
        ax.set(xticks=[1, 2, 3], xlabel='Training seed', ylabel=title); ax.grid(axis='y', alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('H2O initial vs model_200 | same motion and evaluation seed 1\nShared nonterminal prefix: %d steps; first episode only' % horizon)
    fig.savefig(output / 'three_seed_evaluation.png', dpi=180); plt.close(fig)
    print('\nTRAIN SEED | STAGE | GLOBAL mm | ROOT-REL mm | FIRST RESET s')
    for row in results:
        print(row['training_seed'], '|', row['stage'], '|', round(row['global_mpjpe_mm'], 3), '|',
              round(row['root_relative_mpjpe_mm'], 3), '|', row['first_reset_time_s'])

def main():
    root = Path('/content/hr17_three_seed_evaluation')
    output = root / datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    python = Path('/content/hr17_env/bin/python')
    repo = Path('/content/hr17_repo')
    status = dict(status='preparing', training_seeds=[1, 2, 3], evaluation_seed=1,
                  checkpoints=['model_initial.pt', 'model_200.pt'], training_performed=False,
                  maximum_control_steps_per_case=300)
    try:
        assert python.is_file() and repo.is_dir(), 'Restore the environment first'
        pointer = Path('/content/hr17_three_seeds/latest_archive.txt')
        assert pointer.is_file(), 'Missing three-seed training index'
        group = Path(pointer.read_text().strip()).with_suffix('')
        training_status = load(group / 'experiment_status.json')
        assert training_status['status'] == 'completed'
        runs = {int(seed): Path(path) for seed, path in training_status['runs'].items()}
        assert sorted(runs) == [1, 2, 3]
        baseline = runs[1]
        first_metadata = load(baseline / 'run_metadata.json')
        base_cfg = load(baseline / 'config_requested.json')
        import copy
        source_hashes = {}
        for seed, run in sorted(runs.items()):
            assert run.is_dir(), 'Restore run directory: ' + str(run)
            metadata = load(run / 'run_metadata.json')
            assert metadata['status'] == 'completed' and metadata['seed'] == seed
            assert metadata['iterations'] == 200 and metadata['num_envs'] == 128
            assert sha(run / 'model_200.pt') == metadata['final_checkpoint_sha256']
            cfg = load(run / 'config_requested.json')
            expected = copy.deepcopy(base_cfg); expected['seed'] = seed; expected['train']['seed'] = seed
            assert cfg == expected
            for stage, filename in [('initial', 'model_initial.pt'), ('final', 'model_200.pt')]:
                source_hashes['seed%d_%s' % (seed, stage)] = sha(run / filename)
        robot_source = repo / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        before_source_hash = sha(robot_source)
        assert 'quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.clear)' not in robot_source.read_text(), 'Run gravity fix cell 18 first'
        assert capture(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).strip() == first_metadata['repo_commit']
        diff = capture(['git', 'diff', '--no-ext-diff'], cwd=str(repo))
        assert diff.splitlines() == (baseline / 'repo_changes.patch').read_text().splitlines(), 'Repo differs from training'
        env = os.environ.copy()
        env.update(MAX_JOBS='2', MPLBACKEND='Agg', PYTHONHASHSEED='1', PYTHONUNBUFFERED='1')
        env['PATH'] = '/content/hr17_env/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = '/content/hr17_env/lib:/content/isaacgym/python/isaacgym/_bindings/linux-x86_64:' + env.get('LD_LIBRARY_PATH', '')
        freeze = capture([str(python), '-m', 'pip', 'freeze'], env=env)
        assert lines(freeze) == lines((baseline / 'environment_freeze.txt').read_text()), 'Environment differs from training'
        (output / 'environment_freeze.txt').write_text(freeze)
        (output / 'repo_diff.patch').write_text(diff)
        (output / 'gpu_info.txt').write_text(capture(['nvidia-smi']))
        shutil.copy2(group / 'run_index.json', output / 'training_run_index.json')
        shutil.copy2(group / 'training_comparison.json', output / 'source_training_comparison.json')
        shutil.copy2(__file__, output / Path(__file__).name)
        child = output / 'evaluate_case.py'
        child.write_text(DIAGNOSTIC_SOURCE, encoding='utf-8')
        write(output / 'source_checkpoint_hashes.json', source_hashes)
        status['status'] = 'evaluating'
        for seed, stage, filename in CASES:
            label = 'seed%d_%s' % (seed, stage)
            print('\nEVALUATING:', label, '|', filename, '| eval seed 1', flush=True)
            command = [str(python), '-u', str(child), '--checkpoint', filename,
                       '--run-dir', str(runs[seed]), '--output', str(output / label), '--max-steps', '300']
            with (output / (label + '.log')).open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=str(repo), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line)
                code = process.wait()
            assert code == 0, label + ' failed; see saved log'
        summarize(output, source_hashes)
        for seed, stage, filename in CASES:
            assert sha(runs[seed] / filename) == source_hashes['seed%d_%s' % (seed, stage)]
        assert sha(robot_source) == before_source_hash
        status.update(status='completed', source_checkpoint_files_unchanged=True, repository_source_unchanged=True)
        print('\nTHREE-SEED EVALUATION COMPLETED', flush=True)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / 'experiment_status.json', status)
        manifest = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob('*') if path.is_file()}
        write(output / 'SHA256_manifest.json', manifest)
        archive = shutil.make_archive(str(output), 'zip', root_dir=str(output))
        with zipfile.ZipFile(archive) as package: assert package.testzip() is None
        (root / 'latest_archive.txt').write_text(archive)
        print('RESULT ZIP:', archive, flush=True)

if __name__ == '__main__':
    main()

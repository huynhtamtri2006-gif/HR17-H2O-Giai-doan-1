"""Short fresh H2O training on the verified Colab environment.

This is a simplified single-motion experiment, not a full paper reproduction.
No policy or optimizer checkpoint is loaded. Source configs are reused as
explicit hyperparameters, then saved with all overrides and provenance.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import isaacgym
from isaacgym import gymapi, gymutil
import numpy as np
import torch
from easydict import EasyDict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.helpers import set_seed
from rsl_rl.runners import OnPolicyRunner


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f'Cannot serialize {type(item)}')
    path.write_text(json.dumps(value, indent=2, default=convert), encoding='utf-8')


def command_output(arguments, cwd=None):
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'{arguments}: {result.stderr}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--config-dir', type=Path,
                        default=Path('/content/hr17_checkpoint_comparison_input'))
    args = parser.parse_args()
    assert args.num_envs >= 4, 'Use at least 4 environments for the PPO minibatches'
    assert args.iterations > 0

    repo = Path('/content/hr17_repo')
    os.chdir(repo)
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert np.__version__ == '1.23.0', 'Unexpected NumPy version'
    assert torch.__version__ == '2.4.1+cu121', 'Unexpected PyTorch version'
    for filename in ('env_cfg.json', 'train_cfg.json'):
        assert (args.config_dir / filename).exists(), f'Missing {filename}'
    robot_source = repo / 'legged_gym/legged_gym/envs/base/legged_robot.py'
    assert 'self.clear)' not in robot_source.read_text(), 'Apply the documented gravity-vector fix first'

    bangkok = timezone(timedelta(hours=7))
    now = datetime.now(bangkok)
    run_dir = Path('/content/hr17_training') / f'H2O_seed{args.seed}_{now:%Y%m%d_%H%M%S}'
    run_dir.mkdir(parents=True, exist_ok=False)
    Path('/content/hr17_training/latest_run.txt').write_text(str(run_dir))
    shutil.copy2(Path(__file__).resolve(), run_dir / 'hr17_train_fresh.py')
    shutil.copy2(args.config_dir / 'env_cfg.json', run_dir / 'source_env_cfg.json')
    shutil.copy2(args.config_dir / 'train_cfg.json', run_dir / 'source_train_cfg.json')

    cfg = EasyDict(json.loads((args.config_dir / 'env_cfg.json').read_text()))
    cfg.train = EasyDict(json.loads((args.config_dir / 'train_cfg.json').read_text()))
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.num_envs = args.num_envs
    cfg.env.num_envs = args.num_envs
    cfg.headless = True
    cfg.sim_device = cfg.rl_device = 'cuda:0'
    cfg.use_gpu = cfg.use_gpu_pipeline = True
    cfg.test = cfg.env.test = cfg.env.im_eval = False
    cfg.play = False
    cfg.resume = cfg.train.runner.resume = False
    cfg.load_run = cfg.train.runner.load_run = ''
    cfg.train.distill = False
    cfg.train.dagger.dagger_only = False
    cfg.train.dagger.dagger_anneal = False
    cfg.train.runner.has_eval = False
    cfg.max_iterations = cfg.train.runner.max_iterations = args.iterations
    cfg.train.runner.save_interval = 50
    cfg.use_wandb = False
    cfg.train_velocity_estimation = cfg.use_velocity_estimation = False
    cfg.viewer.debug_viz = False
    cfg.motion.visualize = cfg.motion.realtime_vr_keypoints = False
    assert (cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions) == (138, 215, 19)
    assert cfg.motion.teleop_obs_version == 'v-teleop-extend-max'

    resources = repo / 'legged_gym/resources'
    cfg.asset.file = str(resources / 'robots/h1/urdf/h1.urdf')
    cfg.motion.motion_file = str(resources / 'motions/h1/stable_punch.pkl')
    cfg.motion.skeleton_file = str(resources / 'robots/h1/xml/h1.xml')
    cfg.motion.marker_file = str(resources / 'objects/Marker/traj_marker.urdf')
    for path in (cfg.asset.file, cfg.motion.motion_file, cfg.motion.skeleton_file):
        assert Path(path).exists(), f'Missing {path}'

    cfg.terrain.curriculum = False
    cfg.terrain.mesh_type = 'trimesh'
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.border_size = 5
    cfg.terrain.terrain_types = ['flat']
    cfg.terrain.terrain_proportions = [1.0]
    cfg.noise.add_noise = False
    for key, value in list(cfg.domain_rand.items()):
        if key.startswith('randomize_') and isinstance(value, bool):
            cfg.domain_rand[key] = False
    assert isinstance(cfg.domain_rand.randomize_link_body_names, list)
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.motion_package_loss = False
    cfg.domain_rand.born_offset = False
    cfg.domain_rand.born_heading_randomization = False
    cfg.motion.resample_motions_for_envs = False
    # Reward/curriculum coefficients remain those in the saved source config.

    metadata = dict(
        status='preparing', started_at_bangkok=now.isoformat(),
        purpose='fresh short PPO training, simplified single-motion H2O, not full paper reproduction',
        initialization='random; no policy or optimizer checkpoint loaded',
        source_config_provenance='archived run 26_10_01_08-20-15_; original provenance unknown',
        seed=args.seed, num_envs=args.num_envs, iterations=args.iterations,
        actor_observations=138, critic_observations=215, actions=19,
        planned_rollout_transitions=args.num_envs * args.iterations * cfg.train.runner.num_steps_per_env,
        evaluation_performed=False,
        python=sys.version, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__,
        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        motion_sha256=hashlib.sha256(Path(cfg.motion.motion_file).read_bytes()).hexdigest(),
        source_config_sha256=hashlib.sha256((args.config_dir / 'env_cfg.json').read_bytes()).hexdigest(),
        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run_directory=str(run_dir),
        checkpoint_note='model_initial.pt precedes PPO updates; model_0.pt is saved after the first update; final model_200.pt follows 200 updates when iterations=200.',
    )
    metadata['repo_commit'] = command_output(['git', 'rev-parse', 'HEAD'], cwd=repo).strip()
    (run_dir / 'repo_changes.patch').write_text(command_output(['git', 'diff'], cwd=repo))
    (run_dir / 'environment_freeze.txt').write_text(command_output([sys.executable, '-m', 'pip', 'freeze']))
    (run_dir / 'gpu_info.txt').write_text(command_output(['nvidia-smi']))
    for name in ('SMPLSim', 'smplx'):
        dependency = Path('/content/hr17_deps') / name
        if (dependency / '.git').exists():
            (run_dir / f'{name}_commit.txt').write_text(command_output(['git', 'rev-parse', 'HEAD'], cwd=dependency))
            (run_dir / f'{name}_changes.patch').write_text(command_output(['git', 'diff'], cwd=dependency))
    write_json(run_dir / 'config_requested.json', cfg)
    write_json(run_dir / 'run_metadata.json', metadata)

    set_seed(args.seed)
    params = gymapi.SimParams()
    gymutil.parse_sim_config(cfg.sim, params)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    env = runner = None
    start = time.perf_counter()
    print('RUN DIRECTORY:', run_dir, flush=True)
    print('NEW RANDOM POLICY | seed:', args.seed, '| envs:', args.num_envs,
          '| PPO updates:', args.iterations, flush=True)
    try:
        env = LeggedRobot(cfg=cfg, sim_params=params, physics_engine=gymapi.SIM_PHYSX,
                          sim_device='cuda:0', headless=True)
        runner = OnPolicyRunner(env, cfg.train, log_dir=str(run_dir), device=env.device)
        write_json(run_dir / 'env_cfg.json', cfg)
        write_json(run_dir / 'train_cfg.json', cfg.train)
        runner.save(str(run_dir / 'model_initial.pt'))
        metadata['status'] = 'training'
        write_json(run_dir / 'run_metadata.json', metadata)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final_path = run_dir / f'model_{args.iterations}.pt'
        assert final_path.exists(), 'Final checkpoint was not saved'
        metadata.update(status='completed', final_checkpoint=str(final_path),
                        final_checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest())
        print('\nHUAN LUYEN NGAN HOAN TAT', flush=True)
        print('Final checkpoint:', final_path, flush=True)
        print('Training success is not motion-tracking evaluation success.', flush=True)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        metadata['elapsed_wall_seconds'] = time.perf_counter() - start
        metadata['finished_at_bangkok'] = datetime.now(bangkok).isoformat()
        write_json(run_dir / 'run_metadata.json', metadata)
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()

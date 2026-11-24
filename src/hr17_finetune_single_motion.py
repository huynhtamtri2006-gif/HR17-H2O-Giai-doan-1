"""Colab: 100-update single-clip fine-tuning pilot and deterministic evaluation."""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

CHECKPOINT_SHA='59c0bc4d25d8a3d5b468016ad1081fdfc2c44f41899ad4b530bc9573513d28fc'
ROBOT_SHA='484b1e3f96f71d77b1c74a17bea195b2773e7010a616db37a65f4cfc12e9ba26'
MOTION_IDS=['id01','ood03']
REPEATS=3
GAINS=[('kd2',1.0),('kd1',0.5)]
PREVIOUS_STEPS={'id01':75,'ood03':54}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def extract_child(path):
    t=ast.parse(path.read_text(encoding='utf-8'))
    return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
                and any(isinstance(v,ast.Name) and v.id=='DIAGNOSTIC_SOURCE' for v in n.targets))


TRAIN_SOURCE='"""Single-clip fine-tuning pilot: warm-start actor/critic, fresh optimizer, saved local milestones."""\nimport argparse\nimport hashlib\nimport json\nimport os\nimport platform\nimport shutil\nimport subprocess\nimport sys\nimport time\nimport traceback\nfrom datetime import datetime, timedelta, timezone\nfrom pathlib import Path\n\nimport isaacgym\nfrom isaacgym import gymapi, gymutil\nimport numpy as np\nimport torch\nfrom easydict import EasyDict\nfrom legged_gym.envs import LeggedRobot\nfrom legged_gym.utils.helpers import set_seed\nfrom rsl_rl.runners import OnPolicyRunner\n\n\ndef write_json(path, value):\n    def convert(item):\n        if isinstance(item, np.generic):\n            return item.item()\n        if isinstance(item, Path):\n            return str(item)\n        raise TypeError(f\'Cannot serialize {type(item)}\')\n    path.write_text(json.dumps(value, indent=2, default=convert), encoding=\'utf-8\')\n\n\ndef command_output(arguments, cwd=None):\n    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True)\n    if result.returncode != 0:\n        raise RuntimeError(f\'{arguments}: {result.stderr}\')\n    return result.stdout\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\'--seed\', type=int, default=1)\n    parser.add_argument(\'--num-envs\', type=int, default=128)\n    parser.add_argument(\'--iterations\', type=int, default=100)\n    parser.add_argument(\'--output-dir\', type=Path, required=True)\n    parser.add_argument(\'--config-dir\', type=Path,\n                        default=Path(\'/content/hr17_checkpoint_comparison_input\'))\n    args = parser.parse_args()\n    assert args.num_envs >= 4, \'Use at least 4 environments for the PPO minibatches\'\n    assert args.iterations == 100, \'Fixed pilot budget: 100 updates\'\n\n    repo = Path(\'/content/hr17_repo\')\n    os.chdir(repo)\n    assert torch.cuda.is_available(), \'CUDA is unavailable\'\n    assert np.__version__ == \'1.23.0\', \'Unexpected NumPy version\'\n    assert torch.__version__ == \'2.4.1+cu121\', \'Unexpected PyTorch version\'\n    for filename in (\'env_cfg.json\', \'train_cfg.json\'):\n        assert (args.config_dir / filename).exists(), f\'Missing {filename}\'\n    robot_source = repo / \'legged_gym/legged_gym/envs/base/legged_robot.py\'\n    assert \'self.clear)\' not in robot_source.read_text(), \'Apply the documented gravity-vector fix first\'\n\n    bangkok = timezone(timedelta(hours=7))\n    now = datetime.now(bangkok)\n    run_dir = args.output_dir\n    run_dir.mkdir(parents=True, exist_ok=False)\n    # Preserve the original training/latest_run.txt pointer.\n    shutil.copy2(Path(__file__).resolve(), run_dir / \'hr17_train_fresh.py\')\n    shutil.copy2(args.config_dir / \'env_cfg.json\', run_dir / \'source_env_cfg.json\')\n    shutil.copy2(args.config_dir / \'train_cfg.json\', run_dir / \'source_train_cfg.json\')\n\n    cfg = EasyDict(json.loads((args.config_dir / \'env_cfg.json\').read_text()))\n    cfg.train = EasyDict(json.loads((args.config_dir / \'train_cfg.json\').read_text()))\n    cfg.seed = args.seed\n    cfg.train.seed = args.seed\n    cfg.num_envs = args.num_envs\n    cfg.env.num_envs = args.num_envs\n    cfg.headless = True\n    cfg.sim_device = cfg.rl_device = \'cuda:0\'\n    cfg.use_gpu = cfg.use_gpu_pipeline = True\n    cfg.test = cfg.env.test = cfg.env.im_eval = False\n    cfg.play = False\n    cfg.resume = cfg.train.runner.resume = False\n    cfg.load_run = cfg.train.runner.load_run = \'\'\n    cfg.train.distill = False\n    cfg.train.dagger.dagger_only = False\n    cfg.train.dagger.dagger_anneal = False\n    cfg.train.runner.has_eval = False\n    cfg.max_iterations = cfg.train.runner.max_iterations = args.iterations\n    cfg.train.runner.save_interval = 1000000\n    cfg.train.algorithm.learning_rate = 0.0001\n    cfg.train.algorithm.schedule = \'fixed\'\n    cfg.control.damping.ankle = 1.0\n    cfg.use_wandb = False\n    cfg.train_velocity_estimation = cfg.use_velocity_estimation = False\n    cfg.viewer.debug_viz = False\n    cfg.motion.visualize = cfg.motion.realtime_vr_keypoints = False\n    assert (cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions) == (138, 215, 19)\n    assert cfg.motion.teleop_obs_version == \'v-teleop-extend-max\'\n\n    resources = repo / \'legged_gym/resources\'\n    cfg.asset.file = str(resources / \'robots/h1/urdf/h1.urdf\')\n    cfg.motion.motion_file = \'/content/hr17_motion_id_ood/id01.pkl\'\n    cfg.motion.skeleton_file = str(resources / \'robots/h1/xml/h1.xml\')\n    cfg.motion.marker_file = str(resources / \'objects/Marker/traj_marker.urdf\')\n    for path in (cfg.asset.file, cfg.motion.motion_file, cfg.motion.skeleton_file):\n        assert Path(path).exists(), f\'Missing {path}\'\n\n    cfg.terrain.curriculum = False\n    cfg.terrain.mesh_type = \'trimesh\'\n    cfg.terrain.num_rows = cfg.terrain.num_cols = 1\n    cfg.terrain.max_init_terrain_level = 0\n    cfg.terrain.border_size = 5\n    cfg.terrain.terrain_types = [\'flat\']\n    cfg.terrain.terrain_proportions = [1.0]\n    cfg.noise.add_noise = False\n    for key, value in list(cfg.domain_rand.items()):\n        if key.startswith(\'randomize_\') and isinstance(value, bool):\n            cfg.domain_rand[key] = False\n    assert isinstance(cfg.domain_rand.randomize_link_body_names, list)\n    cfg.domain_rand.push_robots = False\n    cfg.domain_rand.motion_package_loss = False\n    cfg.domain_rand.born_offset = False\n    cfg.domain_rand.born_heading_randomization = False\n    cfg.motion.resample_motions_for_envs = False\n    # Reward/curriculum coefficients remain those in the saved source config.\n\n    metadata = dict(\n        status=\'preparing\', started_at_bangkok=now.isoformat(),\n        purpose=\'single-motion fine-tune pilot\',\n        initialization=\'actor and critic loaded from verified model_50; fresh optimizer; local update count reset to zero\',\n        source_config_provenance=\'archived run 26_10_01_08-20-15_; original provenance unknown\',\n        seed=args.seed, num_envs=args.num_envs, iterations=args.iterations,\n        actor_observations=138, critic_observations=215, actions=19,\n        planned_rollout_transitions=args.num_envs * args.iterations * cfg.train.runner.num_steps_per_env,\n        evaluation_performed=False,\n        python=sys.version, platform=platform.platform(),\n        torch=torch.__version__, numpy=np.__version__,\n        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),\n        motion_sha256=hashlib.sha256(Path(cfg.motion.motion_file).read_bytes()).hexdigest(),\n        source_config_sha256=hashlib.sha256((args.config_dir / \'env_cfg.json\').read_bytes()).hexdigest(),\n        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n        run_directory=str(run_dir),\n        checkpoint_note=\'ft_000/025/050/100.pt are saved after exactly 0/25/50/100 local updates. Original archive iteration counter is not interpreted as exact historical update count.\',\n        starting_checkpoint_sha256=\'59c0bc4d25d8a3d5b468016ad1081fdfc2c44f41899ad4b530bc9573513d28fc\',\n        optimizer_loaded=False, ankle_kd=1.0, learning_rate=0.0001, schedule=\'fixed\', training_motion_count=1,\n    )\n    metadata[\'repo_commit\'] = command_output([\'git\', \'rev-parse\', \'HEAD\'], cwd=repo).strip()\n    (run_dir / \'repo_changes.patch\').write_text(command_output([\'git\', \'diff\'], cwd=repo))\n    (run_dir / \'environment_freeze.txt\').write_text(command_output([sys.executable, \'-m\', \'pip\', \'freeze\']))\n    (run_dir / \'gpu_info.txt\').write_text(command_output([\'nvidia-smi\']))\n    for name in (\'SMPLSim\', \'smplx\'):\n        dependency = Path(\'/content/hr17_deps\') / name\n        if (dependency / \'.git\').exists():\n            (run_dir / f\'{name}_commit.txt\').write_text(command_output([\'git\', \'rev-parse\', \'HEAD\'], cwd=dependency))\n            (run_dir / f\'{name}_changes.patch\').write_text(command_output([\'git\', \'diff\'], cwd=dependency))\n    write_json(run_dir / \'config_requested.json\', cfg)\n    write_json(run_dir / \'run_metadata.json\', metadata)\n\n    set_seed(args.seed)\n    params = gymapi.SimParams()\n    gymutil.parse_sim_config(cfg.sim, params)\n    params.use_gpu_pipeline = True\n    params.physx.use_gpu = True\n    env = runner = None\n    start = time.perf_counter()\n    print(\'RUN DIRECTORY:\', run_dir, flush=True)\n    print(\'SINGLE-CLIP FINE-TUNING | seed:\', args.seed, \'| envs:\', args.num_envs,\n          \'| PPO updates:\', args.iterations, flush=True)\n    try:\n        env = LeggedRobot(cfg=cfg, sim_params=params, physics_engine=gymapi.SIM_PHYSX,\n                          sim_device=\'cuda:0\', headless=True)\n        assert env._motion_lib._num_unique_motions == 1\n        ids = [env.dof_names.index(n) for n in [\'left_ankle_joint\', \'right_ankle_joint\']]\n        assert torch.equal(env.d_gains[ids], torch.ones_like(env.d_gains[ids]))\n        runner = OnPolicyRunner(env, cfg.train, log_dir=str(run_dir), device=env.device)\n        start_checkpoint = Path(\'/content/hr17_training/H2O_seed1_20261116_005911/model_50.pt\')\n        assert hashlib.sha256(start_checkpoint.read_bytes()).hexdigest() == metadata[\'starting_checkpoint_sha256\']\n        runner.load(str(start_checkpoint), load_optimizer=False)\n        original_model = torch.load(start_checkpoint, map_location=env.device)[\'model_state_dict\']\n        for key, value in runner.alg.actor_critic.state_dict().items():\n            assert torch.equal(value, original_model[key]), key\n        del original_model\n        runner.current_learning_iteration = 0\n        write_json(run_dir / \'env_cfg.json\', cfg)\n        write_json(run_dir / \'train_cfg.json\', cfg.train)\n        runner.save(str(run_dir / \'model_initial.pt\'))\n        runner.save(str(run_dir / \'ft_000.pt\'))\n        metadata[\'status\'] = \'training\'\n        write_json(run_dir / \'run_metadata.json\', metadata)\n        for milestone in [25, 50, 75, 100]:\n            runner.learn(num_learning_iterations=25, init_at_random_ep_len=False)\n            assert runner.current_learning_iteration == milestone\n            if milestone in [25, 50, 100]:\n                runner.save(str(run_dir / (\'ft_%03d.pt\' % milestone)))\n        final_path = run_dir / f\'model_{args.iterations}.pt\'\n        assert final_path.exists(), \'Final checkpoint was not saved\'\n        metadata.update(status=\'completed\', final_checkpoint=str(final_path),\n                        final_checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),\n                        final_checkpoint_name=final_path.name,\n                        milestone_sha256={name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest()\n                            for name in [\'ft_000.pt\',\'ft_025.pt\',\'ft_050.pt\',\'ft_100.pt\']})\n        print(\'\\nHUAN LUYEN NGAN HOAN TAT\', flush=True)\n        print(\'Final checkpoint:\', final_path, flush=True)\n        print(\'Training success is not motion-tracking evaluation success.\', flush=True)\n    except BaseException as error:\n        metadata.update(status=\'interrupted\' if isinstance(error, KeyboardInterrupt) else \'failed\',\n                        error=repr(error), traceback=traceback.format_exc())\n        raise\n    finally:\n        metadata[\'elapsed_wall_seconds\'] = time.perf_counter() - start\n        metadata[\'finished_at_bangkok\'] = datetime.now(bangkok).isoformat()\n        write_json(run_dir / \'run_metadata.json\', metadata)\n        if runner is not None and runner.writer is not None:\n            runner.writer.flush()\n            runner.writer.close()\n        if env is not None:\n            env.gym.destroy_sim(env.sim)\n\n\nif __name__ == \'__main__\':\n    main()\n'

def capture(command, **kwargs):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=90, **kwargs)
    if result.returncode != 0:
        raise RuntimeError('Command failed: '+repr(command)+'\n'+result.stdout)
    return result.stdout

def make_eval(source):
    source=source.replace("choices=['model_50.pt']", "choices=['ft_000.pt','ft_025.pt','ft_050.pt','ft_100.pt']")
    old="assert metadata['seed'] in [1, 2, 3] and metadata['iterations'] == 200"
    assert old in source
    source=source.replace(old,"assert metadata['seed'] == 1 and metadata['iterations'] == 100 and metadata['purpose'] == 'single-motion fine-tune pilot'")
    old="assert hashlib.sha256((folder / 'model_200.pt').read_bytes()).hexdigest() == metadata['final_checkpoint_sha256']"
    assert old in source
    source=source.replace(old,"assert hashlib.sha256((folder / metadata['final_checkpoint_name']).read_bytes()).hexdigest() == metadata['final_checkpoint_sha256']")
    source=source.replace("intervention_mode='none'","intervention_mode='single_motion_finetune', ankle_kd=1.0")
    source=source.replace('fixed model_50, intact single motion, first episode until termination or motion end; not paper evaluation',
        'single-motion fine-tuning pilot milestone evaluation; training clip only, not generalization or paper evaluation')
    marker="    obs = env.get_observations()"
    assert source.count(marker)==1
    check="    ids = [env.dof_names.index(n) for n in ['left_ankle_joint', 'right_ankle_joint']]\n    assert torch.equal(env.d_gains[ids], torch.ones_like(env.d_gains[ids]))\n"
    source=source.replace(marker,check+marker)
    ast.parse(source)
    return source

def run_logged(cmd,log,env,repo,timeout):
    import threading
    print('RUNNING:', ' '.join(cmd),flush=True)
    with log.open('w') as stream:
        proc=subprocess.Popen(cmd,cwd=str(repo),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        watchdog=threading.Timer(timeout,proc.kill)
        watchdog.daemon=True
        watchdog.start()
        try:
            # Training emits progress each update; outer notebook also streams this output.
            for line in proc.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=proc.wait(timeout=timeout)
        except BaseException:
            proc.kill();proc.wait();raise
        finally:
            watchdog.cancel()
    if code:raise RuntimeError('Process failed; saved log: '+str(log))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print('PLAN: verified model_50 actor+critic, fresh optimizer, 128 envs, single clip id01, ankle Kd1.')
        print('100 updates maximum; fixed LR 1e-4; evaluate 0/25/50/100 on id01 with Kd1.')
        print('This changes training task/gain/optimizer setup; not an isolated causal ablation or full-paper reproduction.')
        return
    base=Path('/content');repo=base/'hr17_repo'
    out=base/'hr17_focus_training'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    original=base/'hr17_training/H2O_seed1_20261116_005911'
    original_checkpoint=original/'model_50.pt'
    robot=repo/'legged_gym/legged_gym/envs/base/legged_robot.py'
    status=dict(status='preparing',purpose='bounded single-clip fine-tuning pilot',no_training=False,
        starting_checkpoint_sha256=CHECKPOINT_SHA,seed=1,num_envs=128,additional_updates=100,
        planned_transitions=128*24*100,learning_rate=0.0001,schedule='fixed',ankle_kd=1,
        optimizer='new Adam state; do not load optimizer from model_50',motion_id='id01',
        evaluation_milestones=[0,25,50,100],initial_lift_m=.04,termination_unchanged=True,
        limit='Training-clip improvement only. Multiple setup changes; not evidence of a single causal factor or unseen-motion generalization.')
    write(out/'fixed_plan.json',status)
    error=None
    try:
        assert sha(original_checkpoint)==CHECKPOINT_SHA and sha(robot)==ROBOT_SHA
        runtime=read(base/'hr17_runtime_environment.json');assert runtime.get('native_h2o_import_verified')
        env=os.environ.copy()
        env.update(PATH='/content/hr17_env/bin:'+env.get('PATH',''),LD_LIBRARY_PATH=runtime['LD_LIBRARY_PATH'],
            MAX_JOBS='2',MPLBACKEND='Agg',PYTHONHASHSEED='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
        py=str(base/'hr17_env/bin/python')
        source=extract_child(base/'hr17_evaluate_motion_id_ood.py')
        motion_manifest=read(base/'hr17_motion_id_ood/manifest.json')
        motion=next(m for m in motion_manifest['motions'] if m['id']=='id01')
        motion_path=base/'hr17_motion_id_ood'/motion['file']
        assert sha(motion_path)==motion['sha256']
        lines=lambda s:{l.strip() for l in s.splitlines() if l.strip() and not l.startswith('#')}
        freeze=capture([py,'-m','pip','freeze'],env=env)
        assert lines(freeze)==lines((original/'environment_freeze.txt').read_text())
        diff=capture(['git','diff','--no-ext-diff'],cwd=str(repo))
        assert diff.splitlines()==(original/'repo_changes.patch').read_text().splitlines()
        (out/'environment_freeze.txt').write_text(freeze);(out/'repo_diff.patch').write_text(diff)
        (out/'gpu_info.txt').write_text(capture(['nvidia-smi']))
        write(out/'runtime_environment.json',runtime);write(out/'motion.json',motion)
        shutil.copy2(__file__,out/Path(__file__).name)
        trainer=out/'train_focus_case.py';trainer.write_text(TRAIN_SOURCE,encoding='utf-8')
        evaluator=out/'evaluate_focus_case.py';evaluator.write_text(make_eval(source),encoding='utf-8')
        status['status']='training';write(out/'experiment_status.json',status)
        training=out/'training'
        run_logged([py,'-u',str(trainer),'--seed','1','--num-envs','128','--iterations','100',
            '--config-dir',str(original),'--output-dir',str(training)],out/'training_console.log',env,repo,1800)
        status['status']='evaluating';write(out/'experiment_status.json',status)
        results=[];frames_list=[]
        for milestone in [0,25,50,100]:
            name='ft_%03d.pt'%milestone;case='eval_%03d'%milestone
            run_logged([py,'-u',str(evaluator),'--checkpoint',name,'--run-dir',str(training),
                '--mass-scale','1.0','--output',str(out/case),
                '--max-steps',str(math.ceil(motion['duration_s']/.02)+1),
                '--motion-file',str(motion_path),'--motion-key',motion['motion_key'],
                '--motion-sha256',motion['sha256']],out/(case+'.log'),env,repo,240)
            summary=read(out/case/'summary.json')
            with (out/case/'frames.csv').open(newline='') as f:frames=list(csv.DictReader(f))
            assert all(r['done']=='False' for r in frames[:-1])
            frames_list.append(frames)
            results.append(dict(local_updates=milestone,checkpoint=name,steps=len(frames),
                observed_time_s=float(frames[-1]['motion_time_s']),first_reset_s=summary['first_reset_time_s'],
                reasons=frames[-1]['reasons'],reached_end=summary['reached_motion_end_without_recorded_failure']))
        common=min(sum(r['done']=='False' for r in f) for f in frames_list);assert common>0
        for row,frames in zip(results,frames_list):
            row.update(common_nonterminal_steps=common,
                common_global_mpjpe_mm=sum(float(r['global_mpjpe_mm']) for r in frames[:common])/common,
                common_root_relative_mpjpe_mm=sum(float(r['root_relative_mpjpe_mm']) for r in frames[:common])/common)
        status.update(status='completed',results=results,
            metric_scope='Post-run shortest nonterminal prefix shared by these four checkpoints; same training clip.')
        write(out/'comparison.json',status)
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
        print('\nPILOT RESULTS:',json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        error=e;status.update(status='failed',error=repr(e))
        (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        status['original_checkpoint_unchanged']=sha(original_checkpoint)==CHECKPOINT_SHA if original_checkpoint.is_file() else None
        status['repository_source_unchanged']=sha(robot)==ROBOT_SHA if robot.is_file() else None
        write(out/'experiment_status.json',status)
        checksums={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256_manifest.json'}
        write(out/'SHA256_manifest.json',checksums)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,str(p.relative_to(out)))
        print('RESULT ZIP:',archive,flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()

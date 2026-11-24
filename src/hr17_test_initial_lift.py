"""Colab: isolate initial root-height lift; original repo/checkpoint untouched."""
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
CASES=[('baseline_4cm',.04),('lift_2cm',.02),('lift_6cm',.06),('baseline_4cm_repeat',.04)]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def extract_child(path):
    t=ast.parse(path.read_text(encoding='utf-8'))
    return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
                and any(isinstance(v,ast.Name) and v.id=='DIAGNOSTIC_SOURCE' for v in n.targets))

OVERRIDE='''    def _reset_root_states(self, env_ids):
        # Baseline follows precisely the unmodified original path.
        super()._reset_root_states(env_ids)
        assert self.num_envs == 1 and self.custom_origins and self.cfg.motion.teleop
        assert not self.cfg.motion.visualize
        delta = cli_args.root_lift_m - 0.04
        if abs(delta) > 1e-9:
            self.root_states[env_ids, 2] += delta
            # Commit only root-state change; never alter policy, reward or termination.
            ids = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(ids), len(ids))

'''

AUDIT='''    offset0 = env.env_origins + env.env_origins_init_3Doffset
    reference0 = env._get_state_from_motionlib_cache_trimesh(
        env.motion_ids, env.motion_start_times, offset=offset0)
    expected0 = reference0['root_pos'][0].detach().clone()
    expected0[2] += cli_args.root_lift_m
    assert torch.allclose(env.root_states[0, :3], expected0, atol=1e-5, rtol=1e-5)
    assert torch.allclose(env.dof_pos, reference0['dof_pos'], atol=1e-5, rtol=1e-5)
    assert torch.allclose(env.dof_vel, reference0['dof_vel'], atol=1e-5, rtol=1e-5)
    (output / 'initial_lift_audit.json').write_text(json.dumps(dict(
        requested_lift_m=cli_args.root_lift_m,
        actual_root_pos=env.root_states[0, :3].detach().cpu().tolist(),
        reference_root_pos=reference0['root_pos'][0].detach().cpu().tolist(),
        actual_lift_m=float((env.root_states[0, 2]-reference0['root_pos'][0, 2]).item()),
        dof_pos_and_vel_match_reference=True,
        caution='Initial rigid-body cache is written by reset code; use integrated frames for physical foot positions.',
    ), indent=2))
'''

def make_child(source):
    source=source.replace('from isaacgym import gymapi, gymutil','from isaacgym import gymapi, gymutil, gymtorch')
    marker='cli_args = parser.parse_args()'
    assert source.count(marker)==1
    source=source.replace(marker,"parser.add_argument('--root-lift-m', type=float, choices=[0.02,0.04,0.06], required=True)\n"+marker)
    marker='    def _process_rigid_body_props(self, props, env_id):'
    assert source.count(marker)==1
    source=source.replace(marker,OVERRIDE+marker)
    marker='    obs = env.get_observations()'
    assert source.count(marker)==1
    source=source.replace(marker,AUDIT+marker)
    source=source.replace("intervention_mode='none'", "intervention_mode='initial_root_lift', initial_root_lift_m=cli_args.root_lift_m")
    source=source.replace('fixed model_50, intact single motion, first episode until termination or motion end; not paper evaluation',
        'initial root-height sensitivity, fixed model_50/id01/seed 1; first episode; exploratory, not training or paper evaluation')
    ast.parse(source)
    return source

def capture(cmd,**kwargs):
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90,**kwargs)
    assert r.returncode==0,r.stdout
    return r.stdout

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print('PLAN: model_50, id01, seed 1; initial lift 4cm / 2cm / 6cm / repeat 4cm.')
        print('No training; no modified reset thresholds; stop at first reset or motion end.')
        print('Add --run only in the restored Colab GPU session.')
        return
    base=Path('/content')
    out=base/'hr17_initial_lift'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    status=dict(status='preparing',purpose='initial-height sensitivity to investigate early reset; improvement not assumed',
        cases=CASES,no_training=True,checkpoint_sha256=CHECKPOINT_SHA,evaluation_seed=1,
        motion_id='id01',termination_unchanged=True)
    paths={}
    error=None
    try:
        repo=base/'hr17_repo'
        run=base/'hr17_training/H2O_seed1_20261116_005911'
        checkpoint=run/'model_50.pt'
        robot=repo/'legged_gym/legged_gym/envs/base/legged_robot.py'
        paths={'checkpoint':checkpoint,'robot':robot}
        assert sha(checkpoint)==CHECKPOINT_SHA
        assert sha(robot)==ROBOT_SHA,'Repository is not the previously verified version.'
        source=extract_child(base/'hr17_evaluate_motion_id_ood.py')
        assert 'normalized_motion_keys' in source
        data=base/'hr17_motion_id_ood'
        manifest=read(data/'manifest.json')
        motion=next(m for m in manifest['motions'] if m['id']=='id01')
        motion_path=data/motion['file']
        assert sha(motion_path)==motion['sha256']
        assert motion['motion_key']=='0-KIT_572_punch_right01_poses'
        runtime=read(base/'hr17_runtime_environment.json')
        assert runtime.get('native_h2o_import_verified')
        env=os.environ.copy()
        env.update(PATH='/content/hr17_env/bin:'+env.get('PATH',''),
            LD_LIBRARY_PATH=runtime['LD_LIBRARY_PATH'],MAX_JOBS='2',MPLBACKEND='Agg',
            PYTHONHASHSEED='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
        py=str(base/'hr17_env/bin/python')
        (out/'gpu_info.txt').write_text(capture(['nvidia-smi']),encoding='utf-8')
        lines=lambda s:{l.strip() for l in s.splitlines() if l.strip() and not l.startswith('#')}
        freeze=capture([py,'-m','pip','freeze'],env=env)
        assert lines(freeze)==lines((run/'environment_freeze.txt').read_text()),'Packages differ from training.'
        diff=capture(['git','diff','--no-ext-diff'],cwd=str(repo))
        assert diff.splitlines()==(run/'repo_changes.patch').read_text().splitlines()
        (out/'environment_freeze.txt').write_text(freeze)
        (out/'repo_diff.patch').write_text(diff)
        write(out/'runtime_environment.json',runtime)
        write(out/'motion.json',motion)
        child=out/'initial_lift_case.py'
        child.write_text(make_child(source),encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        status['status']='running';write(out/'experiment_status.json',status)
        results=[];frame_lists=[]
        for name,lift in CASES:
            print('\nRUNNING:',name,'| initial root lift',lift,'m',flush=True)
            log=out/(name+'.log')
            cmd=[py,'-u',str(child),'--checkpoint','model_50.pt','--output',str(out/name),
                '--max-steps',str(math.ceil(motion['duration_s']/.02)+1),
                '--motion-file',str(motion_path),'--motion-key',motion['motion_key'],
                '--motion-sha256',motion['sha256'],'--root-lift-m',str(lift)]
            with log.open('w') as f:
                proc=subprocess.run(cmd,cwd=str(repo),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
            if proc.returncode:
                print(log.read_text(errors='replace')[-14000:],flush=True)
                raise RuntimeError(name+' failed; saved log: '+str(log))
            s=read(out/name/'summary.json')
            assert s['checkpoint_sha256']==CHECKPOINT_SHA and s['actor_weights_unchanged']
            assert s['initial_root_lift_m']==lift
            assert s['reference_distance_termination_threshold_m']==5.0
            with (out/name/'frames.csv').open(newline='') as f:frames=list(csv.DictReader(f))
            assert frames and all(r['done']=='False' for r in frames[:-1])
            frame_lists.append(frames)
            row=dict(case=name,lift_m=lift,steps=len(frames),
                observed_time_s=float(frames[-1]['motion_time_s']),
                first_reset_s=s['first_reset_time_s'],reasons=frames[-1]['reasons'],
                reached_end=s['reached_motion_end_without_recorded_failure'])
            results.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        common=min(sum(r['done']=='False' for r in frames) for frames in frame_lists)
        assert common>0
        for row,frames in zip(results,frame_lists):
            row['common_nonterminal_steps']=common
            row['common_global_mpjpe_mm']=sum(float(r['global_mpjpe_mm']) for r in frames[:common])/common
            row['common_root_relative_mpjpe_mm']=sum(float(r['root_relative_mpjpe_mm']) for r in frames[:common])/common
        # Repeatability measured, never assume GPU executions are bit-identical.
        repeat_same=results[0]['steps']==results[-1]['steps'] and results[0]['reasons']==results[-1]['reasons']
        baseline_match=results[0]['steps']==75 and results[0]['reasons']=='gravity_x'
        status.update(status='completed',results=results,baseline_matches_saved_reset=baseline_match,
            baseline_repeat_same_steps_and_reasons=repeat_same,
            common_window='post-run shortest nonterminal prefix; descriptive, one clip/seed',
            conclusion='Review reset duration, tracking error and foot contacts before choosing a configuration. No guaranteed improvement or generalized stability.')
        write(out/'comparison.json',status)
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        print('\nSUMMARY:',json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        error=e;status.update(status='failed',error=repr(e))
        (out/'error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        status['checkpoint_unchanged']=sha(paths['checkpoint'])==CHECKPOINT_SHA if 'checkpoint' in paths and paths['checkpoint'].is_file() else None
        status['repository_source_unchanged']=sha(paths['robot'])==ROBOT_SHA if 'robot' in paths and paths['robot'].is_file() else None
        write(out/'experiment_status.json',status)
        manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256_manifest.json'}
        write(out/'SHA256_manifest.json',manifest)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,str(p.relative_to(out)))
        print('RESULT ZIP:',archive,flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()

"""Colab: fixed Kd=1 versus Kd=2 validation on five remaining clips."""
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
MOTION_IDS=['id02','id03','ood01','ood02','ood03']
GAINS=[('kd2',1.0),('kd1',0.5)]
PREVIOUS_STEPS={'id02':69,'id03':72,'ood01':40,'ood02':50,'ood03':54}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def extract_child(path):
    t=ast.parse(path.read_text(encoding='utf-8'))
    return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
                and any(isinstance(v,ast.Name) and v.id=='DIAGNOSTIC_SOURCE' for v in n.targets))

GAIN_SETUP="    ankle_names = ['left_ankle_joint', 'right_ankle_joint']\n    ankle_ids = [env.dof_names.index(name) for name in ankle_names]\n    nominal_d_gains = env.d_gains.detach().clone()\n    nominal_p_gains = env.p_gains.detach().clone()\n    assert torch.allclose(nominal_d_gains[ankle_ids], torch.full_like(nominal_d_gains[ankle_ids], 2.0))\n    if cli_args.ankle_kd_scale != 1.0:\n        env.d_gains[ankle_ids] *= cli_args.ankle_kd_scale\n    expected_d_gains = nominal_d_gains.detach().clone()\n    expected_d_gains[ankle_ids] *= cli_args.ankle_kd_scale\n    assert torch.equal(env.d_gains, expected_d_gains)\n    assert torch.equal(env.p_gains, nominal_p_gains)\n"

AUDIT="    reference0 = env._get_state_from_motionlib_cache_trimesh(\n        env.motion_ids, env.motion_start_times,\n        offset=env.env_origins + env.env_origins_init_3Doffset)\n    expected0 = reference0['root_pos'][0].detach().clone()\n    expected0[2] += 0.04\n    assert torch.allclose(env.root_states[0, :3], expected0, atol=1e-5, rtol=1e-5)\n    assert torch.allclose(env.dof_pos, reference0['dof_pos'], atol=1e-5, rtol=1e-5)\n    assert torch.allclose(env.dof_vel, reference0['dof_vel'], atol=1e-5, rtol=1e-5)\n    (output / 'ankle_damping_audit.json').write_text(json.dumps(dict(\n        names=ankle_names, indices=ankle_ids,\n        ankle_kd_scale=cli_args.ankle_kd_scale,\n        original_d_gains=nominal_d_gains.detach().cpu().tolist(),\n        actual_d_gains=env.d_gains.detach().cpu().tolist(),\n        p_gains_unchanged=True, initial_lift_m=0.04,\n        initial_joint_position_and_velocity_match_reference=True,\n        purpose='Damping sensitivity, not identification of an optimal or critically damped gain.'\n    ), indent=2))\n"

def make_child(source):
    source=source.replace('from isaacgym import gymapi, gymutil','from isaacgym import gymapi, gymutil, gymtorch')
    marker='cli_args = parser.parse_args()'
    assert source.count(marker)==1
    source=source.replace(marker,"parser.add_argument('--ankle-kd-scale', type=float, choices=[0.5,1.0,2.0], required=True)\n"+marker)
    marker='try:\n    # Read the actual actor properties'
    assert source.count(marker)==1
    source=source.replace('try:\n    # Read the actual actor properties', 'try:\n'+GAIN_SETUP+'    # Read the actual actor properties')
    marker='    obs = env.get_observations()'
    assert source.count(marker)==1
    source=source.replace(marker,AUDIT+marker)
    source=source.replace("intervention_mode='none'", "intervention_mode='ankle_damping', ankle_kd_scale=cli_args.ankle_kd_scale")
    source=source.replace('fixed model_50, intact single motion, first episode until termination or motion end; not paper evaluation',
        'ankle-damping sensitivity, fixed model_50/id01/seed 1; first episode; exploratory, not training or paper evaluation')
    assert source.count('ankle_names =')==1
    assert source.count('ankle_damping_audit.json')==1
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
        print('FIXED PLAN: id02/id03/ood01/ood02/ood03; Kd 2 then Kd 1 per clip; 10 rollouts.')
        print('Kd 1 was selected using id01 only. Exploratory cross-motion validation.')
        print('No training; initial lift 4cm; original termination; first episode only.')
        return
    base=Path('/content')
    out=base/'hr17_ankle_validation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    status=dict(status='preparing',purpose='validate fixed ankle Kd1 against nominal Kd2 on remaining motions',
        selection='candidate selected after id01 test; remaining five clips fixed before this test',
        motion_ids=MOTION_IDS,gains=GAINS,no_training=True,checkpoint_sha256=CHECKPOINT_SHA,
        evaluation_seed=1,initial_lift_m=.04,termination_unchanged=True,
        scope='five clips, one checkpoint/evaluation seed; exploratory, not paper evaluation')
    write(out/'fixed_plan.json',status)
    checkpoint=base/'hr17_training/H2O_seed1_20261116_005911/model_50.pt'
    robot=base/'hr17_repo/legged_gym/legged_gym/envs/base/legged_robot.py'
    error=None;rows=[];pairs=[]
    try:
        repo=base/'hr17_repo';run=checkpoint.parent
        assert sha(checkpoint)==CHECKPOINT_SHA
        assert sha(robot)==ROBOT_SHA,'Restore verified repository; do not edit original controller.'
        source=extract_child(base/'hr17_evaluate_motion_id_ood.py')
        assert 'normalized_motion_keys' in source
        data=base/'hr17_motion_id_ood';manifest=read(data/'manifest.json')
        motions={m['id']:m for m in manifest['motions']}
        for key in MOTION_IDS:
            m=motions[key]
            assert sha(data/m['file'])==m['sha256']
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
        write(out/'motion_manifest.json',manifest)
        child=out/'ankle_validation_case.py'
        child.write_text(make_child(source),encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        status['status']='running';write(out/'experiment_status.json',status)
        for key in MOTION_IDS:
            motion=motions[key];frame_pair=[];pair_rows=[]
            for label,scale in GAINS:
                name=key+'_'+label
                print('\nRUNNING:',name,'| ankle Kd=',2*scale,flush=True)
                log=out/(name+'.log')
                cmd=[py,'-u',str(child),'--checkpoint','model_50.pt','--output',str(out/name),
                    '--max-steps',str(math.ceil(motion['duration_s']/.02)+1),
                    '--motion-file',str(data/motion['file']),'--motion-key',motion['motion_key'],
                    '--motion-sha256',motion['sha256'],'--ankle-kd-scale',str(scale)]
                with log.open('w') as f:
                    proc=subprocess.run(cmd,cwd=str(repo),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
                if proc.returncode:
                    print(log.read_text(errors='replace')[-14000:],flush=True)
                    raise RuntimeError(name+' failed; see saved log')
                s=read(out/name/'summary.json')
                assert s['checkpoint_sha256']==CHECKPOINT_SHA and s['actor_weights_unchanged']
                assert s['ankle_kd_scale']==scale and s['reference_distance_termination_threshold_m']==5.0
                assert s['motion_key']==motion['motion_key']
                with (out/name/'frames.csv').open(newline='') as f:frames=list(csv.DictReader(f))
                assert frames and all(r['done']=='False' for r in frames[:-1])
                assert len(frames)<=math.ceil(motion['duration_s']/.02)+1
                frame_pair.append(frames)
                row=dict(case=name,motion_id=key,group=motion['group'],ankle_kd=2*scale,
                    steps=len(frames),observed_time_s=float(frames[-1]['motion_time_s']),
                    first_reset_s=s['first_reset_time_s'],reasons=frames[-1]['reasons'],
                    reached_motion_end=s['reached_motion_end_without_recorded_failure'])
                pair_rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            common=min(sum(r['done']=='False' for r in f) for f in frame_pair)
            assert common>0
            for row,frames in zip(pair_rows,frame_pair):
                row.update(common_pair_nonterminal_steps=common,
                    common_pair_global_mpjpe_mm=sum(float(r['global_mpjpe_mm']) for r in frames[:common])/common,
                    common_pair_root_relative_mpjpe_mm=sum(float(r['root_relative_mpjpe_mm']) for r in frames[:common])/common)
                rows.append(row)
            original,candidate=pair_rows
            delta=candidate['observed_time_s']-original['observed_time_s']
            pair=dict(motion_id=key,common_nonterminal_steps=common,
                baseline_steps_match_saved=original['steps']==PREVIOUS_STEPS[key] and original['reasons']=='gravity_x',
                kd2_time_s=original['observed_time_s'],kd1_time_s=candidate['observed_time_s'],
                kd1_minus_kd2_time_s=delta,
                kd1_duration_result='longer' if delta>.01 else ('shorter' if delta<-.01 else 'same'),
                kd2_global_mm=original['common_pair_global_mpjpe_mm'],kd1_global_mm=candidate['common_pair_global_mpjpe_mm'],
                kd2_root_relative_mm=original['common_pair_root_relative_mpjpe_mm'],kd1_root_relative_mm=candidate['common_pair_root_relative_mpjpe_mm'],
                kd2_reached_end=original['reached_motion_end'],kd1_reached_end=candidate['reached_motion_end'])
            pairs.append(pair);write(out/'completed_pairs.json',pairs)
            print('PAIR:',json.dumps(pair,ensure_ascii=False),flush=True)
        status.update(status='completed',results=rows,pairs=pairs,
            duration_counts={label:sum(p['kd1_duration_result']==label for p in pairs) for label in ['longer','same','shorter']},
            metric_scope='Pair-specific shortest nonterminal prefix, chosen after runs; do not compare absolute means across motions.',
            warning='No automatic next gain search. Judge duration and tracking together; no full-motion claim without end evidence.')
        write(out/'comparison.json',status)
        for filename,data_rows in [('comparison.csv',rows),('paired_results.csv',pairs)]:
            with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
        print('\nDURATION COUNTS:',status['duration_counts'],flush=True)
        print('COMPLETED: 5 motion pairs / 10 rollouts; no training.',flush=True)
    except Exception as e:
        error=e;status.update(status='failed',error=repr(e),completed_pairs=pairs)
        (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        status['checkpoint_unchanged']=sha(checkpoint)==CHECKPOINT_SHA if checkpoint.is_file() else None
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

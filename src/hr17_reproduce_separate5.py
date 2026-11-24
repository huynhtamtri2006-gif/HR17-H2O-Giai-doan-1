"""Colab: 5-update PPO diagnostics, original losses and clipping preserved."""
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



def capture(command, **kwargs):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=90, **kwargs)
    if result.returncode != 0:
        raise RuntimeError('Command failed: '+repr(command)+'\n'+result.stdout)
    return result.stdout

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

import io
import numpy as np

SOURCE_SHA='cd03c9c87a42f5296687af445fc8a5c5bc822fd7e0fa69de0d7c774415ab5c24'
TRAINER_SHA='aab8f3f2a663dc820b63bc961fb48630110a691fa0d4b1db91718472bbb6f017'
EVALUATOR_SHA='5aea50e4019965675d45df05dd8e06f3265c6f31fbb90275e9777cd752ed8b97'

def compare_npz_bytes(left,right):
    with np.load(io.BytesIO(left),allow_pickle=False) as a, np.load(io.BytesIO(right),allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        results={}
        for key in a.files:
            x,y=a[key],b[key]
            same_shape=x.shape==y.shape
            results[key]=dict(shape_left=list(x.shape),shape_right=list(y.shape),
                dtype_left=str(x.dtype),dtype_right=str(y.dtype),
                exact=bool(x.dtype==y.dtype and np.array_equal(x,y)),
                max_abs_difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if same_shape and x.size else None)
        return dict(all_arrays_exact=all(r['exact'] for r in results.values()),arrays=results)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-zip',type=Path,default=Path('/content/hr17_ppo_clipping_comparison/20261124_085613_561254.zip'))
    args=parser.parse_args()
    if not args.run:
        print('PLAN: two independent processes, each 5 updates, exact archived separate-clipping trainer and evaluator.')
        print('Same model50, seed1, id01, Kd1, LR1e-4 fixed, fresh optimizer; total 10 updates / 30,720 transitions.')
        print('Train first; evaluate 0 and 5 only after trainer exits. No changes to child source.')
        print('Same-seed training reruns; not independent seeds. Original preceding joint arm is not replayed.')
        return
    base=Path('/content');repo=base/'hr17_repo'
    out=base/'hr17_reproduce_separate5'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    original=base/'hr17_training/H2O_seed1_20261116_005911'
    checkpoint=original/'model_50.pt';robot=repo/'legged_gym/legged_gym/envs/base/legged_robot.py'
    status=dict(status='preparing',purpose='repeat exact archived separate-clipping five-update child code',
        source_archive_sha256=SOURCE_SHA,trainer_sha256=TRAINER_SHA,evaluator_sha256=EVALUATOR_SHA,
        training_reruns=2,updates_per_rerun=5,total_planned_updates=10,total_planned_transitions=30720,
        no_training=False,
        checkpoint_sha256=CHECKPOINT_SHA,seed=1,num_envs=128,ankle_kd=1,learning_rate=.0001,schedule='fixed',
        optimizer='fresh per rerun, same archived behavior',no_child_source_modifications=True,
        evaluation_schedule='After each trainer process fully exits; before/after checkpoint evaluation only.',
        limitations='Same seed, not independent seeds. Repeats the separate arm code/config, not the full earlier joint-then-separate process/GPU history. No automatic training extension or checkpoint adoption.')
    write(out/'fixed_plan.json',status)
    error=None;results=[];frame_lists=[];comparisons=[]
    try:
        assert args.source_zip.is_file(), 'Upload audited 20261124_085613_561254.zip and pass --source-zip /content/20261124_085613_561254.zip'
        assert sha(args.source_zip)==SOURCE_SHA
        assert sha(checkpoint)==CHECKPOINT_SHA and sha(robot)==ROBOT_SHA
        runtime=read(base/'hr17_runtime_environment.json')
        assert runtime.get('native_h2o_import_verified')
        env=os.environ.copy()
        env.update(PATH='/content/hr17_env/bin:'+env.get('PATH',''),LD_LIBRARY_PATH=runtime['LD_LIBRARY_PATH'],
            MAX_JOBS='2',MPLBACKEND='Agg',PYTHONHASHSEED='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
        py=str(base/'hr17_env/bin/python')
        manifest=read(base/'hr17_motion_id_ood/manifest.json')
        motion=next(m for m in manifest['motions'] if m['id']=='id01')
        motion_path=base/'hr17_motion_id_ood'/motion['file']
        assert sha(motion_path)==motion['sha256']
        source=zipfile.ZipFile(args.source_zip)
        assert source.testzip() is None
        trainer=out/'train_clipping_case.py';trainer.write_bytes(source.read('train_clipping_case.py'))
        evaluator=out/'evaluate_clipping_case.py';evaluator.write_bytes(source.read('evaluate_clipping_case.py'))
        assert sha(trainer)==TRAINER_SHA and sha(evaluator)==EVALUATOR_SHA
        # Compare source configs, not merely the resulting learning-rate fields.
        for name in ['env_cfg.json','train_cfg.json']:
            expected=source.read('separate/training/source_'+name)
            assert sha(original/name)==hashlib.sha256(expected).hexdigest(), 'Source configuration differs: '+name
            (out/('archived_source_'+name)).write_bytes(expected)
        lines=lambda s:{l.strip() for l in s.splitlines() if l.strip() and not l.startswith('#')}
        freeze=capture([py,'-m','pip','freeze'],env=env)
        assert lines(freeze)==lines(source.read('separate/training/environment_freeze.txt').decode())
        diff=capture(['git','diff','--no-ext-diff'],cwd=str(repo))
        assert diff.splitlines()==source.read('separate/training/repo_changes.patch').decode().splitlines()
        (out/'environment_freeze.txt').write_text(freeze);(out/'repo_diff.patch').write_text(diff)
        (out/'gpu_info.txt').write_text(capture(['nvidia-smi']))
        write(out/'runtime_environment.json',runtime);write(out/'motion.json',motion)
        shutil.copy2(__file__,out/Path(__file__).name)
        # Reference rows are archived outcomes, not new runs.
        for updates in [0,5]:
            p='separate/eval_%03d/'%updates
            frames=list(csv.DictReader(io.StringIO(source.read(p+'frames.csv').decode())))
            summary=json.loads(source.read(p+'summary.json'))
            frame_lists.append(frames)
            results.append(dict(run='reference_archive',updates=updates,steps=len(frames),
                observed_time_s=float(frames[-1]['motion_time_s']),reset_s=summary['first_reset_time_s'],
                reasons=frames[-1]['reasons'],reached_end=summary['reached_motion_end_without_recorded_failure']))
        for repeat in [1,2]:
            branch=out/('rerun%d'%repeat);branch.mkdir()
            status.update(status='training',current_rerun=repeat,completed_evaluations=results)
            write(out/'experiment_status.json',status)
            training=branch/'training'
            run_logged([py,'-u',str(trainer),'--seed','1','--num-envs','128','--iterations','5',
                '--config-dir',str(original),'--output-dir',str(training),'--clip-mode','separate'],
                branch/'training_console.log',env,repo,600)
            metadata=read(training/'run_metadata.json')
            assert metadata['status']=='completed' and metadata['clipping_mode']=='separate'
            assert metadata['diagnostic_observed_updates']==5 and metadata['diagnostic_observed_minibatches']==100
            assert metadata['training_script_sha256']==TRAINER_SHA
            for name in ['env_cfg.json','train_cfg.json']:
                assert read(training/name)==json.loads(source.read('separate/training/'+name)), 'Effective config differs: '+name
            raw_comparison=dict(rerun=repeat,against='archived separate arm',rollouts=[])
            for update in range(1,6):
                name='rollout_update_%02d.npz'%update
                compared=compare_npz_bytes((training/name).read_bytes(),source.read('separate/training/'+name))
                raw_comparison['rollouts'].append(dict(update=update,**compared))
            raw_comparison['minibatch_tensors']=compare_npz_bytes(
                (training/'ppo_minibatch_tensors.npz').read_bytes(),source.read('separate/training/ppo_minibatch_tensors.npz'))
            comparisons.append(raw_comparison)
            write(out/'raw_tensor_comparisons.json',comparisons)
            for updates in [0,5]:
                filename='ft_%03d.pt'%updates;case='eval_%03d'%updates
                run_logged([py,'-u',str(evaluator),'--checkpoint',filename,'--run-dir',str(training),
                    '--mass-scale','1.0','--output',str(branch/case),
                    '--max-steps',str(math.ceil(motion['duration_s']/.02)+1),
                    '--motion-file',str(motion_path),'--motion-key',motion['motion_key'],
                    '--motion-sha256',motion['sha256']],branch/(case+'.log'),env,repo,240)
                summary=read(branch/case/'summary.json')
                assert summary['checkpoint_sha256']==sha(training/filename) and summary['actor_weights_unchanged']
                assert summary['reference_distance_termination_threshold_m']==5
                with (branch/case/'frames.csv').open(newline='') as stream:
                    frames=list(csv.DictReader(stream))
                assert frames and all(f['done']=='False' for f in frames[:-1])
                frame_lists.append(frames)
                row=dict(run='rerun%d'%repeat,updates=updates,steps=len(frames),
                    observed_time_s=float(frames[-1]['motion_time_s']),reset_s=summary['first_reset_time_s'],
                    reasons=frames[-1]['reasons'],reached_end=summary['reached_motion_end_without_recorded_failure'])
                results.append(row);print('RESULT:',json.dumps(row),flush=True)
            status['completed_evaluations']=results;write(out/'experiment_status.json',status)
        # Pairwise raw comparison between the two new reruns.
        between=dict(against='rerun1 versus rerun2',rollouts=[])
        for update in range(1,6):
            name='rollout_update_%02d.npz'%update
            compared=compare_npz_bytes((out/'rerun1/training'/name).read_bytes(),(out/'rerun2/training'/name).read_bytes())
            between['rollouts'].append(dict(update=update,**compared))
        between['minibatch_tensors']=compare_npz_bytes(
            (out/'rerun1/training/ppo_minibatch_tensors.npz').read_bytes(),
            (out/'rerun2/training/ppo_minibatch_tensors.npz').read_bytes())
        comparisons.append(between);write(out/'raw_tensor_comparisons.json',comparisons)
        common=min(sum(f['done']=='False' for f in frames) for frames in frame_lists)
        assert common>0 and len(results)==6
        for row,frames in zip(results,frame_lists):
            row.update(common_nonterminal_steps=common,
                global_mpjpe_mm=sum(float(f['global_mpjpe_mm']) for f in frames[:common])/common,
                root_relative_mpjpe_mm=sum(float(f['root_relative_mpjpe_mm']) for f in frames[:common])/common)
        source.close()
        status.update(status='completed',results=results,actual_training_updates=10,
            actual_training_transitions=30720,raw_tensor_comparisons_saved=True,
            metric_scope='Post-run shortest nonterminal prefix across archived 0/5 and both new 0/5 evaluations. Archived rows are not additional new trials.')
        write(out/'comparison.json',status)
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        print('COMPLETED: two exact-child-code training reruns, 5 updates each.',flush=True)
    except Exception as e:
        error=e;status.update(status='failed',error=repr(e),completed_evaluations=results)
        (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        status['original_checkpoint_unchanged']=sha(checkpoint)==CHECKPOINT_SHA if checkpoint.is_file() else None
        status['repository_source_unchanged']=sha(robot)==ROBOT_SHA if robot.is_file() else None
        write(out/'experiment_status.json',status)
        digests={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256_manifest.json'}
        write(out/'SHA256_manifest.json',digests)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as result_zip:
            for p in out.rglob('*'):
                if p.is_file():result_zip.write(p,str(p.relative_to(out)))
        print('RESULT ZIP:',archive,flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()

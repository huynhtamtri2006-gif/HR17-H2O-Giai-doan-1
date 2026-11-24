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

SOURCE_SHA='5c054e8959b82f54b733ff6fe184df5afcb5dbd31f3d1f21276a255b24d8edaf'
BASELINE_SHA='f336544112f29babd7741b459ab16d163bd18299b0be95d4ce199d5f888365bc'
CANDIDATE_SHA='f6441a44044f436a5e5bda99643b357d63c10c74346db2b3bcb8443b502b7392'
EVALUATOR_SHA='5aea50e4019965675d45df05dd8e06f3265c6f31fbb90275e9777cd752ed8b97'
ARMS=[('baseline','ft_000.pt',BASELINE_SHA),('candidate226','ft_005.pt',CANDIDATE_SHA)]

def case_order():
    return [(repeat,arm,filename,digest) for repeat in range(1,6)
        for arm,filename,digest in (ARMS if repeat%2 else list(reversed(ARMS)))]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-zip',type=Path,default=Path('/content/hr17_reproduce_separate5/20261124_101330_883796.zip'))
    args=parser.parse_args()
    if not args.run:
        print('PLAN: id01; fixed original weights and fixed 2.26s candidate; 5 same-seed evaluations each, 10 total.')
        print('No training. Original evaluator bytes, Kd1, original termination. Alternate checkpoint order per pair.')
        print('Assess fixed-checkpoint replay, not reproducibility of training or generalization.')
        return
    base=Path('/content');repo=base/'hr17_repo'
    out=base/'hr17_repeat_candidate_226'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    original=base/'hr17_training/H2O_seed1_20261116_005911/model_50.pt'
    robot=repo/'legged_gym/legged_gym/envs/base/legged_robot.py'
    status=dict(status='preparing',purpose='repeated evaluation of fixed id01 candidate versus original weights',
        no_training=True,planned_evaluations=10,repeats_each=5,evaluation_seed=1,
        baseline_sha256=BASELINE_SHA,candidate_sha256=CANDIDATE_SHA,
        candidate_selection='Best observed id01 checkpoint from rerun1 of archived two-run experiment, selected post hoc.',
        source_archive_sha256=SOURCE_SHA,ankle_kd=1,initial_lift_m=.04,termination_unchanged=True,
        archived_candidate_duration_s=2.26,declared_duration_replay_tolerance_s=.02,
        tolerance_note='One control-step descriptive replay tolerance, not a statistical confidence interval or full-motion success criterion.',
        limitations='Single clip, fixed seed and selected checkpoint. Measures evaluation repeatability only; does not resolve training variability or validate other motions.')
    write(out/'fixed_plan.json',status)
    results=[];frame_lists=[];error=None
    try:
        assert args.source_zip.is_file(), 'Upload 20261124_101330_883796.zip and pass --source-zip /content/20261124_101330_883796.zip'
        assert sha(args.source_zip)==SOURCE_SHA
        assert sha(original)==CHECKPOINT_SHA and sha(robot)==ROBOT_SHA
        runtime=read(base/'hr17_runtime_environment.json')
        assert runtime.get('native_h2o_import_verified')
        env=os.environ.copy()
        env.update(PATH='/content/hr17_env/bin:'+env.get('PATH',''),LD_LIBRARY_PATH=runtime['LD_LIBRARY_PATH'],
            MAX_JOBS='2',MPLBACKEND='Agg',PYTHONHASHSEED='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
        py=str(base/'hr17_env/bin/python')
        run=out/'source_training';run.mkdir()
        with zipfile.ZipFile(args.source_zip) as z:
            hashes=json.loads(z.read('SHA256_manifest.json'))
            for name in ['ft_000.pt','ft_005.pt','model_5.pt','env_cfg.json','train_cfg.json',
                         'run_metadata.json','environment_freeze.txt','repo_changes.patch']:
                member='rerun1/training/'+name;payload=z.read(member)
                assert hashlib.sha256(payload).hexdigest()==hashes[member]
                (run/name).write_bytes(payload)
            evaluator=out/'evaluate_clipping_case.py'
            evaluator.write_bytes(z.read('evaluate_clipping_case.py'))
        assert sha(evaluator)==EVALUATOR_SHA
        assert sha(run/'ft_000.pt')==BASELINE_SHA and sha(run/'ft_005.pt')==CANDIDATE_SHA
        manifest=read(base/'hr17_motion_id_ood/manifest.json')
        motion=next(m for m in manifest['motions'] if m['id']=='id01')
        motion_path=base/'hr17_motion_id_ood'/motion['file']
        assert sha(motion_path)==motion['sha256']
        lines=lambda s:{l.strip() for l in s.splitlines() if l.strip() and not l.startswith('#')}
        freeze=capture([py,'-m','pip','freeze'],env=env)
        assert lines(freeze)==lines((run/'environment_freeze.txt').read_text())
        diff=capture(['git','diff','--no-ext-diff'],cwd=str(repo))
        assert diff.splitlines()==(run/'repo_changes.patch').read_text().splitlines()
        (out/'environment_freeze.txt').write_text(freeze);(out/'repo_diff.patch').write_text(diff)
        (out/'gpu_info.txt').write_text(capture(['nvidia-smi']))
        write(out/'runtime_environment.json',runtime);write(out/'motion.json',motion)
        shutil.copy2(__file__,out/Path(__file__).name)
        for repeat,arm,filename,digest in case_order():
            case=arm+'_r'+str(repeat)
            status.update(status='evaluating',current_case=case,completed_results=results)
            write(out/'experiment_status.json',status)
            print('\nEVALUATING:',case,flush=True)
            run_logged([py,'-u',str(evaluator),'--checkpoint',filename,'--run-dir',str(run),
                '--mass-scale','1.0','--output',str(out/case),
                '--max-steps',str(math.ceil(motion['duration_s']/.02)+1),
                '--motion-file',str(motion_path),'--motion-key',motion['motion_key'],
                '--motion-sha256',motion['sha256']],out/(case+'.log'),env,repo,240)
            summary=read(out/case/'summary.json')
            assert summary['checkpoint_sha256']==digest and summary['actor_weights_unchanged']
            assert summary['motion_key']==motion['motion_key']
            assert summary['reference_distance_termination_threshold_m']==5
            with (out/case/'frames.csv').open(newline='') as stream:frames=list(csv.DictReader(stream))
            assert frames and all(f['done']=='False' for f in frames[:-1])
            row=dict(case=case,arm=arm,repeat=repeat,steps=len(frames),
                observed_time_s=float(frames[-1]['motion_time_s']),first_reset_s=summary['first_reset_time_s'],
                reasons=frames[-1]['reasons'],reached_end=summary['reached_motion_end_without_recorded_failure'])
            results.append(row);frame_lists.append(frames)
            print('RESULT:',json.dumps(row),flush=True)
        assert len(results)==10
        common=min(sum(f['done']=='False' for f in frames) for frames in frame_lists);assert common>0
        for row,frames in zip(results,frame_lists):
            row.update(common_nonterminal_steps=common,
                global_mpjpe_mm=sum(float(f['global_mpjpe_mm']) for f in frames[:common])/common,
                root_relative_mpjpe_mm=sum(float(f['root_relative_mpjpe_mm']) for f in frames[:common])/common)
        paired=[]
        for repeat in range(1,6):
            b=next(r for r in results if r['repeat']==repeat and r['arm']=='baseline')
            c=next(r for r in results if r['repeat']==repeat and r['arm']=='candidate226')
            paired.append(dict(repeat=repeat,baseline_time_s=b['observed_time_s'],
                candidate_time_s=c['observed_time_s'],
                candidate_minus_baseline_s=c['observed_time_s']-b['observed_time_s'],
                baseline_global_mm=b['global_mpjpe_mm'],candidate_global_mm=c['global_mpjpe_mm'],
                baseline_root_relative_mm=b['root_relative_mpjpe_mm'],candidate_root_relative_mm=c['root_relative_mpjpe_mm']))
        durations=[r['observed_time_s'] for r in results if r['arm']=='candidate226']
        assert sha(run/'ft_000.pt')==BASELINE_SHA and sha(run/'ft_005.pt')==CANDIDATE_SHA
        status.update(status='completed',results=results,pairs=paired,checkpoints_unchanged=True,
            candidate_duration_range_s=[min(durations),max(durations)],
            candidate_runs_within_declared_tolerance=sum(abs(t-2.26)<=.020001 for t in durations),
            metric_scope='Common shortest nonterminal prefix across ten rollouts selected post hoc; first episode only.',
            replay_scope='Five reruns of one fixed candidate, not five independent training seeds or an overall success rate.')
        write(out/'comparison.json',status)
        for name,data in [('comparison.csv',results),('paired_results.csv',paired)]:
            with (out/name).open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        print('COMPLETED: 10 evaluation rollouts; no training.',flush=True)
    except Exception as e:
        error=e;status.update(status='failed',error=repr(e),completed_results=results)
        (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        status['original_checkpoint_unchanged']=sha(original)==CHECKPOINT_SHA if original.is_file() else None
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

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import traceback
import zipfile

def varint(data, offset):
    value = shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError('Invalid protobuf varint')
    raise ValueError('Truncated protobuf varint')

def fields(data):
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        wire = key & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
            if len(value) != size:
                raise ValueError('Truncated fixed field')
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
            if len(value) != size:
                raise ValueError('Truncated byte field')
        else:
            raise ValueError(f'Unsupported wire type {wire}')
        yield key >> 3, wire, value

def read_scalars(path):
    with path.open('rb') as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError('Truncated TFRecord header')
            size = struct.unpack('<Q', header[:8])[0]
            if size > 100_000_000:
                raise ValueError('Unexpected TFRecord size')
            payload = stream.read(size)
            if len(payload) != size or len(stream.read(4)) != 4:
                raise ValueError('Truncated TFRecord payload')
            event = list(fields(payload))
            step = next((v for k, w, v in event if k == 2 and w == 0), 0)
            for key, wire, summary in event:
                if key != 5 or wire != 2:
                    continue
                for skey, swire, item in fields(summary):
                    if skey != 1 or swire != 2:
                        continue
                    item_fields = list(fields(item))
                    tag = next((v.decode('utf-8') for k, w, v in item_fields
                                if k == 1 and w == 2), None)
                    number = next((struct.unpack('<f', v)[0] for k, w, v in item_fields
                                   if k == 2 and w == 5), None)
                    if tag is not None and number is not None:
                        yield step, tag, number

TAGS = ['Train/mean_reward', 'Train/mean_episode_length', 'Loss/value_function', 'Policy/mean_noise_std']

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

def validate_requested_config(baseline, current, seed):
    expected = copy.deepcopy(baseline)
    expected['seed'] = seed
    expected['train']['seed'] = seed
    assert current == expected, 'Requested config differs beyond the two seed fields'
    assert current['rewards']['penalty_scale'] == 1.0
    assert current['env']['num_envs'] == 128
    assert current['train']['runner']['max_iterations'] == 200
    assert not current['train']['runner']['resume']

def verify_run(folder, baseline, seed):
    metadata = load(folder / 'run_metadata.json')
    assert metadata['status'] == 'completed' and metadata['seed'] == seed
    assert metadata['num_envs'] == 128 and metadata['iterations'] == 200
    assert metadata['planned_rollout_transitions'] == 614400
    assert metadata['initialization'] == 'random; no policy or optimizer checkpoint loaded'
    assert sha(folder / 'model_200.pt') == metadata['final_checkpoint_sha256']
    assert sha(folder / 'hr17_train_fresh.py') == metadata['training_script_sha256']
    if seed != 1:
        validate_requested_config(load(baseline / 'config_requested.json'),
                                  load(folder / 'config_requested.json'), seed)
        first = load(baseline / 'run_metadata.json')
        for key in ['motion_sha256', 'source_config_sha256', 'training_script_sha256', 'repo_commit',
                    'python', 'torch', 'numpy', 'cuda_runtime', 'gpu']:
            assert metadata[key] == first[key], key
        assert lines((folder / 'environment_freeze.txt').read_text()) == lines((baseline / 'environment_freeze.txt').read_text())
        assert (folder / 'repo_changes.patch').read_text().splitlines() == (baseline / 'repo_changes.patch').read_text().splitlines()
    for name in ['model_initial.pt', 'model_50.pt', 'model_100.pt', 'model_150.pt']:
        assert (folder / name).is_file(), name
    return metadata

def initial_actor_sha(python, folder, env):
    code = ("import torch,hashlib; "
            "c=torch.load(__import__('sys').argv[1],map_location='cpu',weights_only=True); "
            "h=hashlib.sha256(); "
            "[(h.update(k.encode()),h.update(str(tuple(v.shape)).encode()),"
            "h.update(v.detach().cpu().contiguous().numpy().tobytes())) "
            "for k,v in sorted(c['model_state_dict'].items()) if k.startswith('actor.')]; "
            "print(h.hexdigest())")
    output = capture([str(python), '-c', code, str(folder / 'model_initial.pt')], env=env)
    value = output.strip().splitlines()[-1]
    assert len(value) == 64
    return value

def summarize_training(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    statistics_by_seed = {}
    all_values = {}
    for seed, folder in sorted(runs.items()):
        records = []
        for event in sorted(folder.glob('events.out.tfevents.*')):
            records.extend(read_scalars(event))
        assert records and all(math.isfinite(value) for _, _, value in records)
        stats = {}
        selected = {}
        for tag in TAGS:
            values = [(step, value) for step, name, value in records if name == tag]
            assert [step for step, _ in values] == list(range(200)), (seed, tag)
            selected[tag] = values
            stats[tag] = dict(first20_mean=statistics.mean(v for _, v in values[:20]),
                              last20_mean=statistics.mean(v for _, v in values[-20:]))
        statistics_by_seed[str(seed)] = dict(training=stats,
            wall_time_s=load(folder / 'run_metadata.json')['elapsed_wall_seconds'])
        all_values[seed] = selected
        with (output / ('seed%d_training_scalars.csv' % seed)).open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['step', 'tag', 'value']); writer.writerows(records)
    aggregate = {}
    for tag in TAGS:
        values = [statistics_by_seed[str(seed)]['training'][tag]['last20_mean'] for seed in sorted(runs)]
        aggregate[tag] = dict(last20_seed_mean=statistics.mean(values),
                              last20_seed_sample_std=statistics.stdev(values), number_of_seeds=len(values))
    write(output / 'training_comparison.json', dict(
        scope='three independent training seeds of one simplified single-motion configuration; no new policy evaluation',
        seeds=sorted(runs), per_seed=statistics_by_seed, aggregate=aggregate,
        training_completed=True, deterministic_evaluation_performed=False,
        note='Training reward and episode statistics do not establish motion-tracking success.'))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    labels = ['Training reward', 'Training episode length (steps)', 'Value loss', 'Policy action noise std']
    for ax, tag, label in zip(axes.flat, TAGS, labels):
        for seed, values in sorted(all_values.items()):
            points = values[tag]
            ax.plot([x for x, _ in points], [v for _, v in points], label='seed %d' % seed, lw=1, alpha=.85)
        ax.set(xlabel='PPO iteration (0..199)', ylabel=label)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('H2O | seeds 1, 2, 3 | 128 envs | 200 updates | stable_punch\nTraining curves; deterministic policy evaluation remains separate')
    fig.savefig(output / 'three_seed_training_curves.png', dpi=180)
    plt.close(fig)

def package(output, runs):
    target = output.with_suffix('.zip')
    hashes = {}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder in [output] + list(runs.values()):
            prefix = 'audit/' if folder == output else 'runs/' + folder.name + '/'
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts: continue
                name = prefix + path.relative_to(folder).as_posix()
                archive.write(path, name)
                hashes[name] = sha(path)
        archive.writestr('SHA256_manifest.json', json.dumps(hashes, indent=2))
    with zipfile.ZipFile(target) as archive: assert archive.testzip() is None
    (output.parent / 'latest_archive.txt').write_text(str(target))
    print('RESULT ZIP:', target, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', choices=[2, 3], default=[2, 3])
    args = parser.parse_args()
    assert len(args.seeds) == len(set(args.seeds))
    repo = Path('/content/hr17_repo')
    python = Path('/content/hr17_env/bin/python')
    baseline = Path('/content/hr17_training/H2O_seed1_20261116_005911')
    script = baseline / 'hr17_train_fresh.py'
    assert python.is_file() and script.is_file(), 'Restore the baseline training archive first'
    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path('/content/hr17_three_seeds') / timestamp
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    env = os.environ.copy()
    env.update(MAX_JOBS='2', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    env['PATH'] = '/content/hr17_env/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/content/hr17_env/lib:/content/isaacgym/python/isaacgym/_bindings/linux-x86_64:' + env.get('LD_LIBRARY_PATH', '')
    pointer = baseline.parent / 'latest_run.txt'
    pointer_before = pointer.read_text() if pointer.is_file() else str(baseline)
    baseline_files = {str(p): sha(p) for p in baseline.iterdir() if p.is_file()}
    runs = {1: baseline}
    status = dict(status='preparing', reused_seed1=str(baseline), seeds_requested=args.seeds,
                  new_random_initializations=True, num_envs=128, iterations=200,
                  evaluation_performed=False)
    try:
        first = verify_run(baseline, baseline, 1)
        assert sys.version == first['python'], 'Python build differs from seed1; inspect before training'
        runtime = json.loads(capture([str(python), '-c',
            "import torch,numpy,json; print(json.dumps(dict(torch=torch.__version__,"
            "numpy=numpy.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))"], env=env))
        for key in ['torch', 'numpy', 'cuda_runtime', 'gpu']:
            assert runtime[key] == first[key], key
        assert sha(script) == '3274a1f7d6f2f1928e273dae23f64f139ecc81918b911c973637e808663ba466'
        assert sha(baseline / 'source_env_cfg.json') == first['source_config_sha256']
        assert capture(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).strip() == first['repo_commit']
        assert capture(['git', 'diff', '--no-ext-diff'], cwd=str(repo)).splitlines() == (baseline / 'repo_changes.patch').read_text().splitlines(), 'Repo changes differ from seed1'
        for name in ['SMPLSim', 'smplx']:
            dependency = Path('/content/hr17_deps') / name
            assert capture(['git', 'rev-parse', 'HEAD'], cwd=str(dependency)).strip() == (baseline / (name + '_commit.txt')).read_text().strip(), name
            assert capture(['git', 'diff', '--no-ext-diff'], cwd=str(dependency)).splitlines() == (baseline / (name + '_changes.patch')).read_text().splitlines(), name
        expected_freeze = lines((baseline / 'environment_freeze.txt').read_text())
        current_freeze = lines(capture([str(python), '-m', 'pip', 'freeze'], env=env))
        before_fix = sorted(current_freeze)
        if current_freeze != expected_freeze:
            differences = current_freeze.symmetric_difference(expected_freeze)
            assert differences and all(line.lower().startswith('cloudpickle==') for line in differences), 'Environment differs beyond cloudpickle: ' + repr(sorted(differences))
            version = next(line for line in expected_freeze if line.lower().startswith('cloudpickle=='))
            print('Restoring baseline package:', version, flush=True)
            print(capture([str(python), '-m', 'pip', 'install', '--no-deps', version], env=env), flush=True)
            current_freeze = lines(capture([str(python), '-m', 'pip', 'freeze'], env=env))
        assert current_freeze == expected_freeze
        write(output / 'environment_check.json', dict(matches_seed1=True,
            freeze_before=before_fix, freeze_after=sorted(current_freeze)))
        inputs = output / 'source_config'
        inputs.mkdir()
        for name in ['env_cfg.json', 'train_cfg.json']:
            shutil.copy2(baseline / ('source_' + name), inputs / name)
        status['status'] = 'training'
        write(output / 'experiment_status.json', status)
        for seed in args.seeds:
            env['PYTHONHASHSEED'] = str(seed)
            before_dirs = set(baseline.parent.glob('H2O_seed%d_*' % seed))
            command = [str(python), '-u', str(script), '--seed', str(seed),
                       '--num-envs', '128', '--iterations', '200', '--config-dir', str(inputs)]
            print('\nSTART NEW TRAINING SEED:', seed, flush=True)
            with (output / ('seed%d_console.log' % seed)).open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=str(repo), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line)
                code = process.wait()
            created = set(baseline.parent.glob('H2O_seed%d_*' % seed)) - before_dirs
            if len(created) == 1: runs[seed] = next(iter(created))
            assert code == 0 and len(created) == 1, 'Seed %d failed; see console log' % seed
            verify_run(runs[seed], baseline, seed)
            (baseline.parent / ('latest_seed%d_run.txt' % seed)).write_text(str(runs[seed]))
            write(output / 'run_index.json', {str(k): str(v) for k, v in sorted(runs.items())})
            backup = shutil.make_archive(str(runs[seed]), 'zip', root_dir=str(baseline.parent), base_dir=runs[seed].name)
            print('SEED BACKUP ZIP:', backup, flush=True)
        actor_hashes = {str(seed): initial_actor_sha(python, folder, env) for seed, folder in sorted(runs.items())}
        assert len(set(actor_hashes.values())) == len(actor_hashes), 'Initial actor weights unexpectedly identical'
        write(output / 'initial_actor_checks.json', dict(actor_sha256_by_seed=actor_hashes,
                                                       all_initial_actor_weights_different=True))
        if sorted(runs) == [1, 2, 3]:
            summarize_training(runs, output)
        assert all(sha(Path(name)) == expected for name, expected in baseline_files.items()), 'Seed1 files changed'
        status['status'] = 'completed'
        print('\nTHREE-SEED TRAINING COMPLETED', flush=True)
        print('Training completed; motion tracking still requires separate evaluation.', flush=True)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        pointer.write_text(pointer_before)
        status['runs'] = {str(k): str(v) for k, v in sorted(runs.items())}
        write(output / 'experiment_status.json', status)
        package(output, runs)

if __name__ == '__main__':
    main()

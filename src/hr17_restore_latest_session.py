ARCHIVE_HASHES = {'20261118_212729_418901.zip': '702e04e7fdba317adf7cd2be6a1863baa3971e32bf3e92991f73cbd10b448a00', '20261118_215810_217119.zip': 'd170dfaf78cc0fb9beb12436a902793aabd3f5c187244dd35556cb7aa98d7448', '20261118_221112_587011.zip': 'ab86848d750f679aa672d63b40709e0512e622ccaeef8190813a1a9e4fef3589'}
"""Restore verified saved experiments and their environment; never train or evaluate."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile

CONTENT = Path('/content')
PYTHON = CONTENT / 'hr17_env/bin/python'

def sha(data): return hashlib.sha256(data).hexdigest()

def command(args, cwd=None, capture=False):
    result = subprocess.run(args, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    return result.stdout if capture else None

def write_same_or_new(destination, data):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.is_file() and destination.read_bytes() == data, 'Existing file differs: ' + str(destination)
    else:
        destination.write_bytes(data)

def verified_zip(name):
    path = CONTENT / name
    assert path.is_file(), 'Upload: ' + name
    assert sha(path.read_bytes()) == ARCHIVE_HASHES[name], 'ZIP differs from verified original: ' + name
    z = zipfile.ZipFile(path)
    assert z.testzip() is None
    manifest = json.loads(z.read('SHA256_manifest.json'))
    for filename, digest in manifest.items():
        assert sha(z.read(filename)) == digest, filename
    return z

def unpack_member(z, name, root, relative=None):
    member = z.getinfo(name)
    assert not stat.S_ISLNK(member.external_attr >> 16), 'Symlink in ZIP'
    relative = name if relative is None else relative
    target = (root / relative).resolve()
    assert os.path.commonpath([str(root.resolve()), str(target)]) == str(root.resolve())
    if not member.is_dir(): write_same_or_new(target, z.read(name))

def ensure_git(path, url, commit, patch):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): command(['git', 'clone', url, str(path)])
    assert (path / '.git').exists(), 'Expected git repository: ' + str(path)
    current = command(['git', 'rev-parse', 'HEAD'], cwd=str(path), capture=True).strip()
    if current != commit:
        assert not command(['git', 'status', '--porcelain'], cwd=str(path), capture=True).strip(), 'Existing repository has changes'
        command(['git', 'fetch', 'origin', commit], cwd=str(path))
        command(['git', 'checkout', '--detach', commit], cwd=str(path))
    current_diff = command(['git', 'diff', '--no-ext-diff'], cwd=str(path), capture=True)
    if current_diff.splitlines() != patch.splitlines():
        assert not current_diff.strip(), 'Existing repository patch differs: ' + str(path)
        if patch.strip():
            patch_file = CONTENT / ('hr17_restore_' + path.name + '.patch')
            patch_file.write_text(patch, encoding='utf-8')
            command(['git', 'apply', '--check', str(patch_file)], cwd=str(path))
            command(['git', 'apply', str(patch_file)], cwd=str(path))
    assert command(['git', 'rev-parse', 'HEAD'], cwd=str(path), capture=True).strip() == commit
    assert command(['git', 'diff', '--no-ext-diff'], cwd=str(path), capture=True).splitlines() == patch.splitlines()

def main():
    assert sys.version_info[:2] == (3, 8), 'Run with /content/hr17_env/bin/python'
    assert Path('/content/isaacgym/python/setup.py').is_file(), 'Extract Isaac Gym first'
    # Verify all input files before installing or restoring anything.
    packages = {name: verified_zip(name) for name in ARCHIVE_HASHES}
    training = packages['20261118_212729_418901.zip']
    training_status = json.loads(training.read('audit/experiment_status.json'))
    assert training_status['status'] == 'completed'
    baseline = 'runs/H2O_seed1_20261116_005911/'
    metadata = json.loads(training.read(baseline + 'run_metadata.json'))
    ensure_git(CONTENT / 'hr17_repo', 'https://github.com/LeCAR-Lab/human2humanoid.git',
               metadata['repo_commit'], training.read(baseline + 'repo_changes.patch').decode())
    for name in ['SMPLSim', 'smplx']:
        ensure_git(CONTENT / 'hr17_deps' / name, 'https://github.com/ZhengyiLuo/' + name + '.git',
                   training.read(baseline + name + '_commit.txt').decode().strip(),
                   training.read(baseline + name + '_changes.patch').decode())
    freeze = training.read(baseline + 'environment_freeze.txt').decode()
    pinned = [line.strip() for line in freeze.splitlines()
              if re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s]+', line.strip())]
    requirements = CONTENT / 'hr17_saved_requirements.txt'
    requirements.write_text('\n'.join(pinned) + '\n')
    print('RESTORING SAVED PACKAGE VERSIONS:', len(pinned), flush=True)
    command([str(PYTHON), '-m', 'pip', 'install', '--no-deps',
             '--extra-index-url', 'https://download.pytorch.org/whl/cu121', '-r', str(requirements)])
    editable = [CONTENT / 'isaacgym/python', CONTENT / 'hr17_repo/rsl_rl',
                CONTENT / 'hr17_repo/legged_gym', CONTENT / 'hr17_repo/phc',
                CONTENT / 'hr17_deps/SMPLSim', CONTENT / 'hr17_deps/smplx']
    command([str(PYTHON), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation']
            + [part for directory in editable for part in ['-e', str(directory)]])
    actual = command([str(PYTHON), '-m', 'pip', 'freeze'], capture=True)
    lines = lambda s: {line.strip() for line in s.splitlines() if line.strip() and not line.startswith('#')}
    missing, extra = sorted(lines(freeze)-lines(actual)), sorted(lines(actual)-lines(freeze))
    (CONTENT / 'hr17_restore_environment_audit.json').write_text(json.dumps(dict(
        matches_saved_freeze=not missing and not extra, missing_or_different=missing, extra_or_different=extra), indent=2))
    assert not missing and not extra, 'Environment differs; see /content/hr17_restore_environment_audit.json: ' + str((missing, extra))
    group = CONTENT / 'hr17_three_seeds/20261118_212729_418901'
    for name in training.namelist():
        if name.startswith('runs/'):
            unpack_member(training, name, CONTENT / 'hr17_training', name[len('runs/'):])
        elif name.startswith('audit/'):
            unpack_member(training, name, group, name[len('audit/'):])
    for seed, path in training_status['runs'].items():
        folder = Path(path)
        assert folder.parent == CONTENT / 'hr17_training'
        meta = json.loads((folder / 'run_metadata.json').read_text())
        assert meta['seed'] == int(seed) and meta['status'] == 'completed'
        assert sha((folder / 'model_200.pt').read_bytes()) == meta['final_checkpoint_sha256']
    # Pointers require the ZIP alongside its extracted directory.
    shutil.copy2(CONTENT / '20261118_212729_418901.zip', group.with_suffix('.zip'))
    (group.parent / 'latest_archive.txt').write_text(str(group.with_suffix('.zip')))
    for filename, root in [('20261118_215810_217119.zip', 'hr17_three_seed_evaluation'),
                           ('20261118_221112_587011.zip', 'hr17_reward_diagnosis')]:
        z = packages[filename]
        folder = CONTENT / root / Path(filename).stem
        assert json.loads(z.read('experiment_status.json'))['status'] == 'completed'
        for name in z.namelist(): unpack_member(z, name, folder)
        shutil.copy2(CONTENT / filename, folder.with_suffix('.zip'))
        (folder.parent / 'latest_archive.txt').write_text(str(folder.with_suffix('.zip')))
    for filename, archive in [('hr17_evaluate_three_seeds.py', '20261118_215810_217119.zip'),
                               ('hr17_diagnose_rewards.py', '20261118_221112_587011.zip')]:
        write_same_or_new(CONTENT / filename, packages[archive].read(filename))
    baseline_dir = CONTENT / 'hr17_training/H2O_seed1_20261116_005911'
    (baseline_dir.parent / 'latest_run.txt').write_text(str(baseline_dir))
    config_dir = CONTENT / 'hr17_checkpoint_comparison_input'
    for name in ['env_cfg.json', 'train_cfg.json']:
        write_same_or_new(config_dir / name, (baseline_dir / ('source_' + name)).read_bytes())
    env = os.environ.copy()
    env['PATH'] = '/content/hr17_env/bin:' + env.get('PATH', '')
    # Match the earlier successful H2O import check. Do not globally inject
    # bundled Isaac Gym libraries into the dynamic loader search path.
    runtime = Path('/content/hr17_runtime_environment.json')
    saved_runtime = json.loads(runtime.read_text()) if runtime.is_file() else {}
    env['LD_LIBRARY_PATH'] = saved_runtime.get('LD_LIBRARY_PATH') if saved_runtime.get('native_h2o_import_verified') else (
        '/content/hr17_env/lib:' + env.get('LD_LIBRARY_PATH', ''))
    env['PYTHONFAULTHANDLER'] = '1'
    env['MAX_JOBS'] = '2'
    check = '''print("CHECK 1: importing Isaac Gym", flush=True)
import isaacgym
print("CHECK 2: importing PyTorch", flush=True)
import torch, numpy as np
print("CHECK 3: importing H2O", flush=True)
from legged_gym.envs import LeggedRobot
assert torch.__version__ == "2.4.1+cu121" and np.__version__ == "1.23.0"
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("IMPORT MOI TRUONG H2O THANH CONG")
'''
    subprocess.run([str(PYTHON), '-u', '-c', check], cwd='/content/hr17_repo', env=env, check=True)
    for z in packages.values(): z.close()
    print('KHOI PHUC PHIEN MOI THANH CONG — 3 SEED + DANH GIA + CHAN DOAN', flush=True)
    print('NO TRAINING OR EVALUATION WAS RUN', flush=True)

if __name__ == '__main__': main()

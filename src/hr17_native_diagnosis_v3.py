"""Standalone Colab driver/native recovery diagnostic, with conditional backtrace.

Verify a real CUDA driver before creating private libcuda aliases. Never replace
system driver files, Python packages, checkpoints, or repository code. Install
GDB only if ordinary native import still crashes and the debugger is absent.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import shutil
import subprocess
import traceback

CONTENT = Path('/content')
PYTHON = CONTENT / 'hr17_env/bin/python'
REPO = CONTENT / 'hr17_repo'
BINDINGS = CONTENT / 'isaacgym/python/isaacgym/_bindings/linux-x86_64'
ROOT = CONTENT / 'hr17_native_diagnosis_v3' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
RUNTIME = CONTENT / 'hr17_runtime_environment.json'
STATUS = dict(status='preparing', no_training_or_evaluation=True,
              root_cause_determined=False, ordinary_h2o_import_verified=False,
              debugger_installed=False, commands=[])


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def execute(command, filename, env=None, timeout=90, tail=3500, cwd=None):
    print('\nĐANG CHẠY:', filename, '| tối đa', timeout, 'giây', flush=True)
    try:
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        code, output = result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        code, output = 124, output + '\nTIMEOUT after %d seconds\n' % timeout
    (ROOT / filename).write_text(output, encoding='utf-8')
    STATUS['commands'].append(dict(log=filename, return_code=code, timeout_seconds=timeout))
    if tail:
        print(output[-tail:], flush=True)
    print('MÃ KẾT THÚC:', code, flush=True)
    return code, output


DRIVER_PROBE = '''import ctypes, json, sys
driver = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
rc = int(driver.cuInit(0))
count = ctypes.c_int(-1)
count_rc = int(driver.cuDeviceGetCount(ctypes.byref(count))) if rc == 0 else None
result = dict(library=sys.argv[1], cuInit=rc, device_count_code=count_rc, devices=count.value)
print("DRIVER_JSON=" + json.dumps(result), flush=True)
sys.exit(0 if rc == 0 and count_rc == 0 and count.value > 0 else 2)
'''

NATIVE_PROBE = '''print("CHECK 1: ISAAC GYM", flush=True)
import isaacgym
print("CHECK 2: PYTORCH/GPU", flush=True)
import torch, numpy as np
assert torch.__version__ == "2.4.1+cu121", torch.__version__
assert np.__version__ == "1.23.0", np.__version__
assert torch.cuda.is_available()
value = torch.ones((2, 2), device="cuda")
assert (value @ value)[0, 0].item() == 2.0
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("CHECK 3: H2O", flush=True)
from legged_gym.envs import LeggedRobot
print("H2O_NATIVE_V3_OK", flush=True)
'''


def native_check(name, env):
    code, output = execute([str(PYTHON), '-u', '-X', 'faulthandler', str(ROOT/'native_probe.py')],
                           name, env, timeout=120, cwd=str(REPO))
    return code == 0 and 'H2O_NATIVE_V3_OK' in output


def main():
    assert PYTHON.is_file() and REPO.is_dir() and BINDINGS.is_dir(), 'Restore environment/data first'
    assert (CONTENT/'hr17_env/lib/libpython3.8.so.1.0').is_file(), 'Missing Python 3.8 shared library'
    ROOT.mkdir(parents=True, exist_ok=False)
    # Invalidate a stale readiness flag; only ordinary imports below can restore it.
    if RUNTIME.is_file():
        shutil.copy2(RUNTIME, ROOT/'previous_runtime_environment.json')
    save(RUNTIME, dict(native_h2o_import_verified=False,
                       reason='Native import being rechecked', diagnostic_directory=str(ROOT)))
    env = os.environ.copy()
    env.update(PATH='/content/hr17_env/bin:'+env.get('PATH',''), MAX_JOBS='2',
               PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', MPLBACKEND='Agg')
    inherited = env.get('LD_LIBRARY_PATH', '')
    save(ROOT/'host.json', dict(platform=platform.platform(),
        inherited_library_path=inherited, inherited_preload=env.get('LD_PRELOAD'),
        pythonpath=env.get('PYTHONPATH'), time_utc=datetime.now(timezone.utc).isoformat()))
    execute(['nvidia-smi'], 'nvidia_smi.log', env, timeout=30, tail=1500)
    candidates = []
    for directory in ['/usr/lib64-nvidia', '/usr/lib/x86_64-linux-gnu',
                      '/usr/local/nvidia/lib64', '/usr/lib/wsl/lib']:
        folder = Path(directory)
        for path in sorted(folder.glob('libcuda.so*')) if folder.is_dir() else []:
            real = path.resolve()
            if real.is_file() and '/stubs/' not in str(real) and real not in candidates:
                candidates.append(real)
    STATUS['driver_candidates'] = [str(path) for path in candidates]
    selected, driver_result = None, None
    for index, candidate in enumerate(candidates[:8]):
        code, output = execute([str(PYTHON), '-u', '-c', DRIVER_PROBE, str(candidate)],
                               'cuda_driver_%d.log'%index, env, timeout=20, tail=1500)
        if code == 0:
            payload = next((line[len('DRIVER_JSON='):] for line in output.splitlines()
                            if line.startswith('DRIVER_JSON=')), None)
            if payload:
                driver_result = json.loads(payload)
                if driver_result['cuInit']==0 and driver_result['devices']>0:
                    selected = candidate
                    break
    assert selected is not None, 'No CUDA driver passed cuInit/device-count checks; see driver logs'
    links = ROOT/'driver_links'
    links.mkdir()
    for name in ['libcuda.so', 'libcuda.so.1']:
        (links/name).symlink_to(selected)
    env['LD_LIBRARY_PATH'] = ':'.join([str(links), '/content/hr17_env/lib', inherited])
    settings = dict(verified_gpu_driver=True, selected_driver_library=str(selected),
                    driver_probe=driver_result, LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'])
    save(ROOT/'driver_environment.json', settings)
    print('ĐÃ XÁC NHẬN DRIVER CUDA VÀ TẠO LIÊN KẾT RIÊNG:', links, flush=True)
    for name in ['gym_38.so','libcarb.so','libPhysXGpu_64.so']:
        execute(['ldd', str(BINDINGS/name)], 'ldd_'+name+'.log', env, timeout=20, tail=0)
    save(ROOT/'native_file_hashes.json', {name:hashlib.sha256((BINDINGS/name).read_bytes()).hexdigest()
        for name in ['gym_38.so','libcarb.so','libPhysXGpu_64.so','carb.config.json']
        if (BINDINGS/name).is_file()})
    (ROOT/'native_probe.py').write_text(NATIVE_PROBE, encoding='utf-8')
    consecutive = 0
    for number in [1,2]:
        if native_check('ordinary_before_debug_%d.log'%number, env):
            consecutive += 1
        else:
            break
    if consecutive < 2:
        # GDB is a diagnostic tool, not evidence that ordinary import is repaired.
        gdb = shutil.which('gdb')
        if not gdb:
            print('Import thường vẫn lỗi: cài GDB để lấy backtrace; không cài lại Python/Torch.', flush=True)
            apt_env = os.environ.copy()
            apt_env['DEBIAN_FRONTEND'] = 'noninteractive'
            code, _ = execute(['apt-get','update','-qq'], 'apt_update.log', apt_env, timeout=180, tail=1500)
            assert code == 0, 'Cannot prepare GDB; diagnostic ZIP contains installation log'
            code, _ = execute(['apt-get','install','-y','--no-install-recommends','--no-upgrade','gdb'],
                               'apt_gdb.log', apt_env, timeout=240, tail=2000)
            assert code == 0, 'GDB installation failed; send diagnostic ZIP'
            STATUS['debugger_installed'] = True
            gdb = shutil.which('gdb')
        assert gdb, 'GDB executable unavailable'
        command = [gdb,'-q','-nx','--batch',
            '-ex','set pagination off','-ex','set confirm off',
            '-ex','set debuginfod enabled off','-ex','run',
            '-ex','thread apply all bt 30','-ex','info sharedlibrary',
            '--args',str(PYTHON),'-u','-X','faulthandler',str(ROOT/'native_probe.py')]
        _, debugger_output = execute(command, 'gdb_backtrace.log', env, timeout=180,
                                      tail=4500, cwd=str(REPO))
        STATUS['h2o_marker_under_debugger'] = 'H2O_NATIVE_V3_OK' in debugger_output
        print('Đang kiểm tra lại bằng hai tiến trình thông thường, không qua GDB.', flush=True)
        consecutive = 0
        for number in [1,2]:
            if native_check('ordinary_after_debug_%d.log'%number, env):
                consecutive += 1
            else:
                break
    if consecutive == 2:
        runtime = dict(LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'], native_h2o_import_verified=True,
            ordinary_import_checks_passed=2, verified_by='hr17_native_diagnosis_v3',
            diagnostic_directory=str(ROOT), selected_driver_library=str(selected),
            verified_gpu_driver=True)
        save(RUNTIME, runtime)
        os.environ['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
        STATUS.update(status='ordinary_import_verified', ordinary_h2o_import_verified=True)
        print('\nHAI LẦN IMPORT H2O/GPU THÔNG THƯỜNG ĐỀU THÀNH CÔNG.', flush=True)
        print('ĐÃ LƯU /content/hr17_runtime_environment.json — chạy lại ô 2 phép đối chứng.', flush=True)
    else:
        STATUS['status'] = 'native_import_still_failed'
        print('\nIMPORT THÔNG THƯỜNG VẪN LỖI — gửi ZIP chẩn đoán; chưa chạy đối chứng.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        ROOT.mkdir(parents=True, exist_ok=True)
        STATUS.update(status='diagnosis_failed', error=repr(error), traceback=traceback.format_exc())
        print('\nCHƯA XÁC NHẬN ĐƯỢC MÔI TRƯỜNG:', error, flush=True)
    finally:
        if ROOT.is_dir():
            save(ROOT/'status.json', STATUS)
            archive = Path(shutil.make_archive(str(ROOT), 'zip', root_dir=str(ROOT)))
            (ROOT.parent/'latest_archive.txt').write_text(str(archive), encoding='utf-8')
            print('ZIP CHẨN ĐOÁN:', archive, flush=True)
            try:
                from google.colab import files
                files.download(str(archive))
            except ImportError:
                pass

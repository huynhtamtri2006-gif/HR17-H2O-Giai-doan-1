"""Bounded, isolated native-import probes after restore reaches its import check.

Does not install packages, edit the repo, or execute training/evaluation.
Run as a regular Python cell in the Colab notebook kernel, not hr17_env Python.
"""
from pathlib import Path
from datetime import datetime
import json
import os
import shutil
import subprocess

root = Path('/content/hr17_native_recheck') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
root.mkdir(parents=True, exist_ok=False)
python = '/content/hr17_env/bin/python'
assert Path(python).is_file(), 'Chưa có môi trường Python 3.8'
assert Path('/content/hr17_repo').is_dir(), 'Chưa có repo đã khôi phục'
base = os.environ.copy()
base['PATH'] = '/content/hr17_env/bin:' + base.get('PATH', '')
base['MAX_JOBS'] = '2'
base['PYTHONFAULTHANDLER'] = '1'
base['PYTHONUNBUFFERED'] = '1'
base['MPLBACKEND'] = 'Agg'

probe = '''print("CHECK 1: Isaac Gym", flush=True)
import isaacgym
print("CHECK 2: PyTorch", flush=True)
import torch, numpy as np
assert torch.__version__ == "2.4.1+cu121"
assert np.__version__ == "1.23.0"
assert torch.cuda.is_available()
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("CHECK 3: H2O", flush=True)
from legged_gym.envs import LeggedRobot
print("H2O_NATIVE_OK", flush=True)
'''

inherited = base.get('LD_LIBRARY_PATH', '')
# The second test changes only this subprocess's library-search order.
# Existing system directories are tried before the Conda libraries.
system_paths = [p for p in ['/usr/lib64-nvidia', '/usr/local/nvidia/lib64',
                          '/usr/lib/x86_64-linux-gnu'] if Path(p).is_dir()]
candidates = [('colab_inherited', inherited),
              ('system_first', ':'.join(system_paths + ['/content/hr17_env/lib', inherited]))]
status = {'purpose': 'native import diagnosis, no training or evaluation',
          'original_LD_LIBRARY_PATH': inherited,
          'original_LD_PRELOAD': base.get('LD_PRELOAD'), 'cases': [], 'success': False}
for name, loader_path in candidates:
    print('\nĐANG KIỂM TRA:', name, '| tối đa 90 giây', flush=True)
    env = base.copy()
    env['LD_LIBRARY_PATH'] = loader_path
    try:
        result = subprocess.run([python, '-u', '-X', 'faulthandler', '-c', probe],
                                cwd='/content/hr17_repo', env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        code, output = result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        if isinstance(output, bytes): output = output.decode(errors='replace')
        code, output = 124, output + '\nTIMEOUT: 90 seconds\n'
    (root / (name + '.log')).write_text(output, encoding='utf-8')
    print(output[-4500:], flush=True)
    print('MÃ KẾT THÚC:', code, flush=True)
    passed = code == 0 and 'H2O_NATIVE_OK' in output
    status['cases'].append({'name': name, 'return_code': code,
                            'LD_LIBRARY_PATH': loader_path, 'passed': passed})
    if passed:
        runtime = {'LD_LIBRARY_PATH': loader_path, 'native_h2o_import_verified': True,
                   'verified_by': 'hr17_native_recheck', 'diagnostic_directory': str(root)}
        Path('/content/hr17_runtime_environment.json').write_text(
            json.dumps(runtime, indent=2), encoding='utf-8')
        os.environ['LD_LIBRARY_PATH'] = loader_path
        status['success'] = True
        print('KIỂM TRA H2O/GPU THÀNH CÔNG — chạy tiếp ô 13, không chạy lại ô 9.', flush=True)
        break

if not status['success']:
    print('CẢ HAI PHÉP KIỂM TRA CHƯA THÀNH CÔNG — chưa chạy ô 13.', flush=True)
    print('Tải ZIP bên dưới và gửi lại; không cần cài lại thư viện.', flush=True)

(root / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
archive = Path(shutil.make_archive(str(root), 'zip', root_dir=str(root)))
Path('/content/hr17_native_recheck/latest_archive.txt').write_text(str(archive), encoding='utf-8')
print('GÓI LOG:', archive, flush=True)
if not status['success']:
    from google.colab import files
    files.download(str(archive))

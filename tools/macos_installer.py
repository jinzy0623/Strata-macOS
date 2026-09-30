"""One-tool Apple Silicon installation: inspect, rank, verify, benchmark and deploy.

Recommendation means the largest model in the pinned catalog which fits the
live memory budget and passes the chosen local responsiveness profile. It is
not a claim to find the world's best model or a general answer-quality score.
"""
from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import os
import platform
import plistlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import psutil

SOURCE = Path(__file__).resolve().parents[1]
HOME_DIR = Path(os.environ.get('STRATA_HOME', Path.home() / 'Library/Application Support/Strata-macOS'))
GIB = 2**30
LABEL = 'com.jinzy0623.strata-macos'
PROFILES = {'balanced': (15.0, 8.0), 'quality': (6.0, 20.0), 'fast': (30.0, 4.0)}
CATALOG = SOURCE / 'config/model-catalog.json'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def sysctl(key):
    result = subprocess.run(['/usr/sbin/sysctl', '-n', key], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ''


def hardware(directory):
    mem = psutil.virtual_memory()
    return {'chip': sysctl('machdep.cpu.brand_string'), 'architecture': platform.machine(),
            'macos': platform.mac_ver()[0], 'cpu_count': os.cpu_count(),
            'memory_total': mem.total, 'memory_available': mem.available,
            'swap_used': psutil.swap_memory().used, 'disk_free': shutil.disk_usage(directory).free,
            'rosetta': sysctl('sysctl.proc_translated') == '1'}


def required_memory(model, context):
    # Conservatively allow 10% model overhead, FP16 KV and a 768MiB workspace.
    return int(sum(f['bytes'] for f in model['files']) * 1.1 +
               model['kv_bytes_per_token'] * context + 0.75 * GIB)


def rank_models(models, hw, context, *, live=True):
    budget = min(hw['memory_total'] * 0.60, hw['memory_total'] - 3 * GIB)
    if live:
        budget = min(budget, hw['memory_available'] - GIB)
    return sorted((m for m in models if required_memory(m, context) <= budget),
                  key=lambda m: m['parameters'], reverse=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path, spec):
    return path.is_file() and path.stat().st_size == spec['bytes'] and sha256_file(path) == spec['sha256']


def disk_needed(model, directory):
    # Resume partial files, and never require full space again for cached weights.
    needed = 0
    for spec in model['files']:
        final = directory / spec['name']
        if valid_file(final, spec):
            continue
        part = final.with_suffix('.gguf.part')
        needed += max(0, spec['bytes'] - (part.stat().st_size if part.exists() else 0))
    return needed + 2 * GIB


def download_model(model, models_dir):
    directory = models_dir / model['id']
    directory.mkdir(parents=True, exist_ok=True)
    if disk_needed(model, directory) > shutil.disk_usage(directory).free:
        raise RuntimeError('磁盘空间不足；需要模型剩余下载空间以及至少 2GB 余量。')
    for spec in model['files']:
        final = directory / spec['name']
        print(f"  校验 / 下载 {spec['name']}（{spec['bytes']/GIB:.2f} GiB）", flush=True)
        if valid_file(final, spec):
            continue
        part = final.with_suffix('.gguf.part')
        if final.exists():
            # Retain invalid files for diagnosis, outside the model's shard set.
            final.rename(final.with_suffix('.gguf.invalid'))
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{spec['name']}"
        for attempt in range(2):
            command = ['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                       '--speed-time', '120', '--speed-limit', '1024', '--silent', '--show-error',
                       '--continue-at', '-', '--output', str(part), url]
            proc = subprocess.Popen(command, start_new_session=True)
            previous_bytes = part.stat().st_size if part.exists() else 0
            last_note = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic() - last_note >= 10:
                        downloaded = part.stat().st_size if part.exists() else 0
                        seconds = time.monotonic() - last_note
                        print(f"  已下载 {downloaded/spec['bytes']:.0%}，{max(0, downloaded-previous_bytes)/seconds/2**20:.1f} MiB/s", flush=True)
                        previous_bytes, last_note = downloaded, time.monotonic()
                    time.sleep(0.5)
                if proc.returncode:
                    raise RuntimeError('模型下载中断。重新运行安装工具会从已下载部分继续。')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
            if valid_file(part, spec):
                os.replace(part, final)
                break
            part.unlink(missing_ok=True)
            if attempt == 1:
                raise RuntimeError('模型 SHA256 校验失败，未部署该文件。')
            print('  文件校验失败，重新下载一次。', flush=True)
    atomic_json(directory / 'SOURCE.json', model)
    # ggml transparently opens sibling shards, beginning with shard 00001.
    return directory / model['files'][0]['name']


def logged_process(command, logfile, *, cwd=SOURCE, timeout=1800, guard_memory=False):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    baseline_swap = psutil.swap_memory().used
    started = last_note = time.monotonic()
    min_available = psutil.virtual_memory().available
    peak_rss = max_swap = 0
    with logfile.open('w') as log:
        proc = subprocess.Popen([str(c) for c in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                min_available = min(min_available, psutil.virtual_memory().available)
                max_swap = max(max_swap, psutil.swap_memory().used - baseline_swap)
                try:
                    peak_rss = max(peak_rss, psutil.Process(proc.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    raise RuntimeError(f'操作超时（{timeout}s），详情：{logfile}')
                if guard_memory and (psutil.virtual_memory().available < 0.5 * GIB or
                                     psutil.swap_memory().used - baseline_swap > 256 * 2**20):
                    raise RuntimeError('实测时内存余量过低或交换内存明显增加，终止此候选模型。')
                if time.monotonic() - last_note > 20:
                    print(f'  仍在进行，已用 {int(elapsed)} 秒；日志：{logfile.name}', flush=True)
                    last_note = time.monotonic()
                time.sleep(0.25)
            if proc.returncode:
                raise RuntimeError(f'操作未通过（exit {proc.returncode}），详情：{logfile}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

    return {"min_available_bytes": min_available, "swap_growth_bytes": max(0, max_swap),
            "process_peak_rss_bytes": peak_rss}


def ensure_runtime(build_dir, logs):
    probe = subprocess.run([sys.executable, '-c',
        'import importlib.metadata; from llama_cpp import llama_cpp; '
        'assert importlib.metadata.version("llama-cpp-python")=="0.3.16"; '
        'assert llama_cpp.llama_supports_gpu_offload()'], capture_output=True)
    if probe.returncode:
        env = dict(os.environ, CMAKE_ARGS='-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_ARCHITECTURES=arm64',
                   CMAKE_BUILD_PARALLEL_LEVEL='4')
        # Keep environment scoped to the child, without changing the user's shell.
        with (logs / 'metal-build.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--force-reinstall',
                                     '--no-binary', 'llama-cpp-python', '-r', str(SOURCE / 'requirements-macos.txt')],
                                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            build_started = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic() - build_started > 1800:
                        raise RuntimeError("Metal 编译超时，请查看 metal-build.log")
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        print('  正在编译 Metal 推理库…', flush=True)
                if proc.returncode:
                    raise RuntimeError(f'Metal 编译失败，请查看 {logs / "metal-build.log"}')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
    cmake = Path(sys.executable).parent / 'cmake'
    ctest = Path(sys.executable).parent / 'ctest'
    logged_process([cmake, '-S', SOURCE, '-B', build_dir, '-DSTRATA_ENABLE_CUDA=OFF'], logs / 'configure.log')
    logged_process([cmake, '--build', build_dir, '--parallel', '4'], logs / 'native-build.log')
    logged_process([ctest, '--test-dir', build_dir, '--output-on-failure'], logs / 'platform-tests.log')
    logged_process([sys.executable, '-m', 'unittest', 'serve.test_server', 'tests_macos.test_backend',
                    'tests_macos.test_installer', '-v'], logs / 'api-tests.log', timeout=120)


def request_json(base, path, body=None, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def api_checks(base, model_name):
    for path in ('/', '/web/app.js', '/metrics', '/v1/models'):
        with urllib.request.urlopen(base + path, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f'HTTP 验证失败：{path}')
    body = {'model': model_name, 'messages': [{'role': 'user', 'content': 'Say hello in one short sentence.'}],
            'max_tokens': 32, 'temperature': 0, 'seed': 42}
    result = request_json(base, '/v1/chat/completions', body)
    content = result['choices'][0]['message'].get('content', '')
    if not content.strip():
        raise RuntimeError('普通 API 返回了空回答。')
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps({**body, 'stream': True}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        lines = response.read().decode()
    chunks = [json.loads(line[6:]) for line in lines.splitlines() if line.startswith('data: {')]
    if 'data: [DONE]' not in lines or not any(c['choices'][0]['delta'].get('content') for c in chunks):
        raise RuntimeError('流式 API 验证失败。')
    chinese = request_json(base, '/v1/chat/completions', {**body,
        'messages': [{'role': 'user', 'content': '请用中文写一句简短的问候。'}]})
    chinese_text = chinese['choices'][0]['message'].get('content', '')
    if not any('\u4e00' <= char <= '\u9fff' for char in chinese_text):
        raise RuntimeError('中文生成验证失败。')
    arithmetic = request_json(base, '/v1/chat/completions', {**body,
        'messages': [{'role': 'user', 'content': 'What is 17 + 25? Reply with only the number.'}]})
    arithmetic_text = arithmetic['choices'][0]['message'].get('content', '')
    if re.search(r'(?<!\d)42(?!\d)', arithmetic_text) is None:
        raise RuntimeError('基础算术生成验证失败；未将本次验证算作通过。')
    return {'english': content, 'chinese': chinese_text, 'arithmetic': arithmetic_text,
            'web': True, 'nonstream': True, 'stream': True}


def benchmark_worker(config_path, output_path):
    from serve.backends import load_backend
    from serve.server import Service, serve
    config = json.loads(Path(config_path).read_text())
    bundle = load_backend('metal', config)
    svc = Service(bundle.engine, bundle.tokenizer, bundle.template, model_name=config['model_name'])
    svc.stop_ids = bundle.stop_ids
    server = serve(svc, port=0)
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        checks = api_checks(base, config['model_name'])
        trials = []
        # Short and longer prompts, sustained output. Prefix differs between trials
        # so this is not a cached replay of a 0.5B greeting.
        for prefix in ('', ('The local library preserves stories about people, nature and technology. ' * 60)):
            prompt = prefix + 'Write a long original story in English about a traveler building a library. Keep writing several paragraphs.'
            ids, _, cap = svc.prepare([{'role': 'user', 'content': prompt}], None, {}, 128)
            started, first, count = time.monotonic(), None, 0
            for token in bundle.engine.generate(ids, cap, {'temperature': 0, 'seed': 42}, threading.Event()):
                if token in bundle.stop_ids:
                    break
                if first is None:
                    first = time.monotonic()
                count += 1
            ended = time.monotonic()
            if count < 32 or first is None:
                raise RuntimeError('实测输出太短，无法可靠评估生成速度。')
            trials.append({'prompt_tokens': len(ids), 'output_tokens': count,
                           'ttft_s': first - started, 'decode_tokens_s': (count - 1) / max(1e-6, ended - first)})
        result = {'checks': checks, 'trials': trials,
                  'decode_tokens_s': min(t['decode_tokens_s'] for t in trials),
                  'ttft_s': max(t['ttft_s'] for t in trials)}
        atomic_json(Path(output_path), result)
    finally:
        server.shutdown()
        server.server_close()
        bundle.engine.close()


def benchmark(config, logs, profile):
    cfg_path = logs / f"{config['model_name']}.trial-config.json"
    out_path = logs / f"{config['model_name']}.benchmark.json"
    out_path.unlink(missing_ok=True)
    atomic_json(cfg_path, config)
    logfile = logs / f"{config['model_name']}.metal.log"
    memory = logged_process([sys.executable, '-m', 'tools.macos_installer', '--benchmark-worker', cfg_path, out_path],
                   logfile, timeout=240, guard_memory=True)
    result = json.loads(out_path.read_text())
    # Require the real llama.cpp load log, not merely compiled GPU capability.
    result['memory'] = memory
    result['metal_verified'] = 'device Metal' in logfile.read_text(errors='replace')
    min_speed, max_wait = PROFILES[profile]
    result['passed'] = result['metal_verified'] and result['decode_tokens_s'] >= min_speed and result['ttft_s'] <= max_wait
    result['profile'] = {'name': profile, 'min_decode_tokens_s': min_speed, 'max_ttft_s': max_wait}
    return result


def agent_path():
    return Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')


def owned_agent():
    path = agent_path()
    if not path.exists():
        return False
    data = plistlib.loads(path.read_bytes())
    if data.get('WorkingDirectory') != str(HOME_DIR / 'runtime'):
        raise RuntimeError('同名部署属于另一个安装目录；请先用该安装的工具停止服务。')
    return True


def bootout():
    subprocess.run(['launchctl', 'bootout', f'gui/{os.getuid()}/{LABEL}'], capture_output=True)


def pick_port():
    for port in range(8080, 8091):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                pass
    raise RuntimeError('8080–8090 均被占用，请释放一个本地端口。')


def wait_ready(base, installation_id, timeout=90):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            health = request_json(base, '/health', timeout=2)
            if health.get('installation_id') == installation_id and health.get('backend') == 'metal':
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError('部署服务未就绪，请查看 logs/server.err.log；未将安装标为成功。')


def deploy(config, logs):
    runtime = HOME_DIR / 'runtime'
    pending = HOME_DIR / ('runtime.pending-' + uuid.uuid4().hex)
    pending.mkdir(parents=True)
    backup = HOME_DIR / ('runtime.backup-' + uuid.uuid4().hex)
    for name in ('serve', 'tools', 'config'):
        if SOURCE / name != runtime / name:
            shutil.copytree(SOURCE / name, pending / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'macos.json', '*report*', '*state*', '*.shared-settings.json'))
    cfg_path = HOME_DIR / 'config/macos.json'
    previous_config = cfg_path.read_bytes() if cfg_path.exists() else None
    path = agent_path()
    previous_plist = path.read_bytes() if owned_agent() else None
    port = pick_port()
    config = {**config, 'port': port, 'host': '127.0.0.1', 'installation_id': uuid.uuid4().hex}
    atomic_json(cfg_path, config)
    data = {'Label': LABEL, 'ProgramArguments': [sys.executable, '-m', 'serve.server', '--engine', 'metal',
            '--config', str(cfg_path), '--port', str(port)], 'WorkingDirectory': str(runtime), 'RunAtLoad': True,
            'KeepAlive': {'SuccessfulExit': False}, 'ThrottleInterval': 10,
            'StandardOutPath': str(logs / 'server.out.log'), 'StandardErrorPath': str(logs / 'server.err.log')}
    path.parent.mkdir(parents=True, exist_ok=True)
    had_runtime = runtime.exists()
    if had_runtime:
        runtime.rename(backup)
    pending.rename(runtime)
    try:
        path.write_bytes(plistlib.dumps(data))
        result = subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(path)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'后台服务启动失败：{result.stderr.strip()}')
        base = f'http://127.0.0.1:{port}'
        wait_ready(base, config['installation_id'])
        checks = api_checks(base, config['model_name'])
        return {'base_url': base, 'openai_base_url': base + '/v1', 'port': port,
                'installation_id': config['installation_id'], 'checks': checks, 'login_autostart': True}
    except BaseException:
        bootout()
        runtime.rename(HOME_DIR / ('runtime.failed-' + uuid.uuid4().hex))
        if had_runtime:
            backup.rename(runtime)
        if previous_config is None:
            cfg_path.unlink(missing_ok=True)
        else:
            cfg_path.write_bytes(previous_config)
        if previous_plist is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(previous_plist)
            subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(path)], capture_output=True)
        raise


def write_controls():
    for action, filename in (('start', '打开 Strata.command'), ('stop', '停止 Strata.command')):
        path = HOME_DIR / filename
        # Environment paths are shell-quoted, including spaces and user names.
        import shlex
        text = '#!/bin/bash\nexport STRATA_HOME=' + shlex.quote(str(HOME_DIR)) + '\n'
        text += 'cd ' + shlex.quote(str(HOME_DIR / 'runtime')) + '\n'
        text += shlex.quote(sys.executable) + ' -m tools.macos_installer --' + action + '\n'
        path.write_text(text)
        path.chmod(0o755)


def control(action, *, open_browser=True):
    if not owned_agent():
        raise RuntimeError('尚未完成后台部署，请先运行安装工具。')
    config = json.loads((HOME_DIR / 'config/macos.json').read_text())
    base = f"http://127.0.0.1:{config['port']}"
    if action == 'stop':
        bootout()
        print('Strata 已停止；再次打开启动工具即可继续使用。')
    elif action == 'start':
        proc = subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(agent_path())], capture_output=True)
        if proc.returncode:
            # Already running is acceptable only after identity-checked readiness.
            print('  检查现有后台服务…')
        wait_ready(base, config['installation_id'])
        if open_browser:
            subprocess.run(['open', base], check=False)
        print(f'已就绪：{base}')
    else:
        wait_ready(base, config['installation_id'], timeout=3)
        print(f"正在运行：{config['model_name']}，{base}")


def main():
    ap = argparse.ArgumentParser(description='自动推荐、下载、验证并部署 Strata Metal 模型')
    ap.add_argument('--profile', choices=PROFILES, default='balanced')
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--model', help='手动指定目录中的型号，如 qwen2.5-7b；仍须通过内存和实测检查')
    ap.add_argument('--plan', action='store_true', help='只查看设备与候选推荐，不下载模型')
    ap.add_argument('--build-only', action='store_true', help='仅构建及运行代码测试，供 CI 使用；不是完成部署')
    ap.add_argument('--no-open', action='store_true')
    group = ap.add_mutually_exclusive_group()
    for action in ('start', 'stop', 'status'):
        group.add_argument('--' + action, action='store_true')
    ap.add_argument('--benchmark-worker', nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.benchmark_worker:
        benchmark_worker(*args.benchmark_worker)
        return 0
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        ap.error('仅支持原生 Apple Silicon macOS。')
    if int(platform.mac_ver()[0].split('.')[0]) < 14:
        ap.error('需要 macOS 14 或更新版本。')
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    installation_lock = (HOME_DIR / '.installer.lock').open('a')
    try:
        fcntl.flock(installation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('另一个安装或部署操作正在运行，请等待它完成。')
    for action in ('start', 'stop', 'status'):
        if getattr(args, action):
            control(action, open_browser=not args.no_open)
            return 0
    if not 512 <= args.context <= 32768:
        ap.error('context 应在 512–32768 之间。')
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    logs = HOME_DIR / 'logs'
    logs.mkdir(exist_ok=True)
    if args.build_only:
        ensure_runtime(SOURCE / 'build-macos', logs)
        print('构建和代码测试通过。build-only 模式未下载模型或部署服务。')
        return 0
    models = json.loads(CATALOG.read_text())['models']
    if args.model and args.model not in {m['id'] for m in models}:
        ap.error('未知型号，请使用 config/model-catalog.json 中的 id。')
    report = {'schema': 1, 'status': 'incomplete', 'profile': args.profile, 'context': args.context,
              'attempts': [], 'installation_home': str(HOME_DIR)}
    suspended, deployed = False, False
    try:
        if not args.plan and owned_agent():
            bootout()
            suspended = True
            time.sleep(2)
        hw = hardware(HOME_DIR)
        report['hardware'] = hw
        print(f"[1/5] {hw['chip']} · {hw['memory_total']/GIB:.0f}GB 统一内存 · 当前可用 {hw['memory_available']/GIB:.1f}GB", flush=True)
        if hw['rosetta']:
            raise RuntimeError('请在原生 ARM64 环境运行安装工具。')
        candidates = rank_models(models, hw, args.context)
        capacity = rank_models(models, hw, args.context, live=False)
        report['capacity_recommendation'] = capacity[0]['id'] if capacity else None
        if capacity and candidates and capacity[0]['id'] != candidates[0]['id']:
            print(f"  硬件容量可考虑 {capacity[0]['id']}，但当前内存占用较高，先验证 {candidates[0]['id']}。关闭不用的应用后重跑可重新评估。", flush=True)
        if args.model:
            candidates = [m for m in candidates if m['id'] == args.model]
        if not candidates:
            raise RuntimeError('当前内存不足以安全运行目录中的模型。请关闭不用的应用后重新运行；不以 0.5B 冒充日常推荐。')
        report['candidates'] = [m['id'] for m in candidates]
        print('  待验证候选：' + ', '.join(report['candidates']), flush=True)
        if args.plan:
            print('  尚未实测，这只是容量筛选，不是已验证推荐。')
            return 0
        print('[2/5] 安装 Metal 运行环境、构建 ARM64 工具并运行代码测试…', flush=True)
        ensure_runtime(SOURCE / 'build-macos', logs)
        latest = hardware(HOME_DIR)
        candidates = rank_models(models, latest, args.context)
        if args.model:
            candidates = [m for m in candidates if m['id'] == args.model]
        report['hardware_before_benchmark'] = latest
        report['candidates'] = [m['id'] for m in candidates]
        print('[3/5] 下载并校验模型，测量真实 Metal 推理性能…', flush=True)
        print('  按当前内存重新筛选：' + ', '.join(report['candidates']), flush=True)
        selected = result = config = None
        for model in candidates:
            # Recheck memory immediately before each download/load, since other apps can change it.
            if model not in rank_models([model], hardware(HOME_DIR), args.context):
                report['attempts'].append({'model': model['id'], 'passed': False, 'reason': '当前内存不足，跳过'})
                continue
            model_path = download_model(model, HOME_DIR / 'models')
            config = {'model': str(model_path), 'model_name': model['id'], 'context': args.context, 'fit_max_tokens': True}
            try:
                result = benchmark(config, logs, args.profile)
                report['attempts'].append({'model': model['id'], **result})
                print(f"  {model['id']}：{result['decode_tokens_s']:.1f} token/s，最慢首字 {result['ttft_s']:.2f}s", flush=True)
                if result['passed']:
                    selected = model
                    break
                print('  未达到所选响应要求，尝试下一档。', flush=True)
            except RuntimeError as error:
                report['attempts'].append({'model': model['id'], 'passed': False, 'reason': str(error)})
                print(f'  {error} 尝试下一档。', flush=True)
        if selected is None:
            raise RuntimeError('候选模型均未通过本机验证，未部署。可释放内存后重试，或用 --profile quality 放宽速度要求。')
        print('[4/5] 部署本地后台服务，并再次验证网页、普通和流式 API…', flush=True)
        deployment = deploy(config, logs)
        deployed = True
        write_controls()
        report.update(status='complete', selected_model=selected['id'], model_source=selected,
                      benchmark=result, deployment=deployment, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        atomic_json(HOME_DIR / 'config/install-report.json', report)
        print(f"[5/5] 全部完成。推荐并部署：{selected['id']}\n网页：{deployment['base_url']}\nOpenAI：{deployment['openai_base_url']}\n验证报告：{HOME_DIR / 'config/install-report.json'}", flush=True)
        if not args.no_open:
            subprocess.run(['open', deployment['base_url']], check=False)
        return 0
    except BaseException as error:
        report['error'] = str(error)
        atomic_json(HOME_DIR / 'config/install-report.failed.json', report)
        raise
    finally:
        if suspended and not deployed and owned_agent():
            subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(agent_path())], capture_output=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('已中断，安装未完成；重新打开安装工具可继续下载。', file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError) as error:
        print(f'安装未完成：{error}', file=sys.stderr, flush=True)
        sys.exit(1)

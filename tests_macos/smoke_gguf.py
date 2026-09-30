"""Real GGUF integration check; takes a user-provided small instruction GGUF.
Usage: .venv/bin/python -m tests_macos.smoke_gguf /path/model.gguf
Requires an actual Metal build and device. Model files are not committed.
"""
import json
import sys
import urllib.request
from serve.backends import load_backend
from serve.server import Service, serve

bundle = load_backend('metal', {'model': sys.argv[1], 'context': 1024})
svc = Service(bundle.engine, bundle.tokenizer, bundle.template, model_name='smoke-gguf')
svc.stop_ids = bundle.stop_ids
httpd = serve(svc, port=0)
base = f'http://127.0.0.1:{httpd.server_address[1]}'

def post(body):
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=120)

try:
    for path in ('/', '/web/app.js', '/health', '/metrics', '/v1/models'):
        with urllib.request.urlopen(base + path) as response:
            assert response.status == 200
    body = {'model': 'smoke-gguf', 'messages': [{'role': 'user', 'content': 'Say hello in one short sentence.'}],
            'max_tokens': 32, 'temperature': 0, 'seed': 42}
    with post(body) as response:
        result = json.load(response)
    answer = result['choices'][0]['message']['content']
    assert answer and result['usage']['completion_tokens'] > 0, result
    print('NONSTREAM:', answer)
    with post({**body, 'stream': True}) as response:
        lines = response.read().decode()
    assert 'data: [DONE]' in lines and '"content"' in lines, lines
    print('STREAM: OK')
    # Ensure a second request works after an early length stop.
    with post({**body, 'max_tokens': 2}) as response:
        assert json.load(response)['usage']['completion_tokens'] <= 2
    print('WEB, HEALTH, METRICS, MODELS, repeated generation: OK')
finally:
    httpd.shutdown()
    httpd.server_close()
    bundle.engine.close()

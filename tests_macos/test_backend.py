import threading
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from serve.backends import load_backend
from serve.backends.ggml import GGMLBackend, GGMLTemplate, create_backend


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.closed = False
    def n_ctx(self): return 1024
    def generate(self, ids, **kwargs):
        self.calls += 1
        self.options = kwargs
        try:
            yield from range(10)
        finally:
            self.closed = True
    def set_seed(self, seed): self.seed = seed


class BackendTests(unittest.TestCase):
    def test_limit_closes_generator_and_forwards_sampling(self):
        model = FakeModel()
        engine = GGMLBackend(model, metal=False)
        self.assertEqual(list(engine.generate([1], 2, {'temperature': 0, 'seed': 42}, threading.Event())), [0, 1])
        self.assertTrue(model.closed)
        self.assertEqual(model.seed, 42)
        self.assertEqual(model.options['temp'], 0)
        self.assertEqual(engine.last['generated'], 2)

    def test_cancel_before_prefill(self):
        model = FakeModel()
        event = threading.Event()
        event.set()
        self.assertEqual(list(GGMLBackend(model, metal=False).generate([1], 2, {}, event)), [])
        self.assertEqual(model.calls, 0)

    def test_cancel_between_tokens(self):
        model = FakeModel()
        event = threading.Event()
        gen = GGMLBackend(model, metal=False).generate([1], 10, {}, event)
        self.assertEqual(next(gen), 0)
        event.set()
        self.assertEqual(list(gen), [])
        self.assertTrue(model.closed)

    def test_thinking_from_rendered_prompt(self):
        self.assertFalse(GGMLTemplate.starts_in_reasoning('assistant\n'))
        self.assertTrue(GGMLTemplate.starts_in_reasoning('assistant\n<think>\n'))
        self.assertFalse(GGMLTemplate.starts_in_reasoning('<think></think>\n'))

    def test_wrong_architecture_rejected_before_loading(self):
        with patch('serve.backends.ggml.platform.machine', return_value='x86_64'):
            with self.assertRaisesRegex(ValueError, 'native ARM64'):
                create_backend({}, metal=True)

    def test_missing_shard_is_rejected_before_native_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model-00001-of-00002.gguf'
            path.write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, 'Missing GGUF shards'):
                create_backend({'model': str(path)}, metal=False)

    def test_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, 'unknown backend'):
            load_backend('unknown', {})

if __name__ == '__main__':
    unittest.main()

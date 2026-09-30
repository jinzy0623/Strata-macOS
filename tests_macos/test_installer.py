import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import plistlib

from tools.macos_installer import GIB, SOURCE, atomic_json, disk_needed, rank_models, required_memory, valid_file, wait_ready, deploy

MODELS = json.loads((SOURCE / 'config/model-catalog.json').read_text())['models']


class RecommendationTests(unittest.TestCase):
    def hw(self, total, available):
        return {'memory_total': total * GIB, 'memory_available': available * GIB}

    def test_16gb_does_not_recommend_14b_or_toy_05b(self):
        ranked = rank_models(MODELS, self.hw(16, 12), 4096)
        self.assertEqual(ranked[0]['id'], 'qwen2.5-7b')
        self.assertNotIn('qwen2.5-14b', [m['id'] for m in ranked])

    def test_busy_machine_downsizes_from_capacity(self):
        hw = self.hw(16, 4)
        self.assertEqual(rank_models(MODELS, hw, 4096)[0]['id'], 'qwen2.5-1.5b')
        self.assertEqual(rank_models(MODELS, hw, 4096, live=False)[0]['id'], 'qwen2.5-7b')

    def test_high_context_changes_recommendation(self):
        self.assertEqual(rank_models(MODELS, self.hw(24, 16), 4096)[0]['id'], 'qwen2.5-14b')
        self.assertEqual(rank_models(MODELS, self.hw(24, 16), 32768)[0]['id'], 'qwen2.5-7b')

    def test_large_machine_ranks_larger_model(self):
        self.assertEqual(rank_models(MODELS, self.hw(64, 48), 4096)[0]['id'], 'qwen2.5-32b')

    def test_not_enough_memory_does_not_claim_success(self):
        self.assertEqual(rank_models(MODELS, self.hw(8, 2), 4096), [])

    def test_all_shards_count_towards_budget(self):
        model = next(m for m in MODELS if m['id'] == 'qwen2.5-7b')
        self.assertGreater(required_memory(model, 4096), sum(f['bytes'] for f in model['files']))
        self.assertEqual(len(model['files']), 2)


class IntegrityAndDeploymentTests(unittest.TestCase):
    def test_checksum_detects_corruption_with_same_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.gguf'
            path.write_bytes(b'good')
            spec = {'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
            self.assertTrue(valid_file(path, spec))
            path.write_bytes(b'evil')
            self.assertFalse(valid_file(path, spec))

    def test_resume_space_accounts_for_partial_download(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            model = {'files': [{'name': 'm.gguf', 'bytes': 100}]}
            (path / 'm.gguf.part').write_bytes(b'x' * 40)
            self.assertEqual(disk_needed(model, path), 60 + 2 * GIB)

    def test_atomic_json_preserves_complete_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config/a.json'
            atomic_json(path, {'status': 'complete'})
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')
            self.assertFalse(path.with_name('a.json.tmp').exists())

    def test_unrelated_service_cannot_pass_readiness(self):
        with patch('tools.macos_installer.request_json', return_value={'backend': 'metal', 'installation_id': 'someone-else'}):
            with self.assertRaisesRegex(RuntimeError, '未就绪'):
                wait_ready('http://127.0.0.1:8080', 'our-id', timeout=0.01)

    def test_failed_deployment_restores_previous_config_and_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            home = root / 'home'
            for name in ('serve', 'tools', 'config'):
                (source / name).mkdir(parents=True)
            (source / 'serve/new.py').write_text('new version')
            (home / 'runtime').mkdir(parents=True)
            (home / 'runtime/old.txt').write_text('last known good version')
            (home / 'config').mkdir()
            config_path = home / 'config/macos.json'
            old_config = b'{"model": "old-model"}'
            config_path.write_bytes(old_config)
            agent = root / 'agent.plist'
            old_agent = plistlib.dumps({'WorkingDirectory': str(home / 'runtime')})
            agent.write_bytes(old_agent)
            with patch('tools.macos_installer.HOME_DIR', home), patch('tools.macos_installer.SOURCE', source), \
                 patch('tools.macos_installer.agent_path', return_value=agent), \
                 patch('tools.macos_installer.pick_port', return_value=8080), \
                 patch('tools.macos_installer.subprocess.run', return_value=SimpleNamespace(returncode=1, stderr='failed')):
                with self.assertRaisesRegex(RuntimeError, '启动失败'):
                    deploy({'model_name': 'new-model'}, home / 'logs')
            self.assertEqual(config_path.read_bytes(), old_config)
            self.assertEqual(agent.read_bytes(), old_agent)
            self.assertEqual((home / 'runtime/old.txt').read_text(), 'last known good version')
            self.assertFalse((home / 'runtime/serve/new.py').exists())

    def test_our_metal_instance_passes_readiness(self):
        with patch('tools.macos_installer.request_json', return_value={'backend': 'metal', 'installation_id': 'our-id'}):
            wait_ready('http://127.0.0.1:8080', 'our-id', timeout=1)

if __name__ == '__main__':
    unittest.main()

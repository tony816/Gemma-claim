"""Offline setup tests: no credentials, cloud calls, or private fixtures needed."""
import contextlib
import io
import json
import os
import socket
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'serving'))
import client_config
import claim_client
import launch_test


class SetupTests(unittest.TestCase):
    def test_busy_port_starts_ui_on_another_port_without_inference(self):
        sys.path.insert(0, str(ROOT / 'space'))
        import test_ui
        import gradio.http_server as http_server
        real_launch = test_ui.demo.launch
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            busy_port = occupied.getsockname()[1]
            with patch.object(http_server, 'INITIAL_PORT_VALUE', busy_port), \
                 patch.object(sys, 'argv', ['launch_test.py']), \
                 patch.object(test_ui.demo, 'launch',
                              side_effect=lambda **kw: real_launch(prevent_thread_lock=True, **kw)), \
                 patch('webbrowser.open') as browser, \
                 patch.object(claim_client, 'post', side_effect=AssertionError('Paid request forbidden')) as submit:
                try:
                    launch_test.main()
                    self.assertNotEqual(test_ui.demo.server_port, busy_port)
                    self.assertGreater(test_ui.demo.server_port, busy_port)
                    self.assertTrue(browser.called)
                    self.assertEqual(browser.call_args.args[0], test_ui.demo.local_url)
                    with urllib.request.urlopen(test_ui.demo.local_url, timeout=10) as response:
                        self.assertEqual(response.status, 200)
                    submit.assert_not_called()
                finally:
                    test_ui.demo.close()

    def test_env_precedence_and_secret_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, '.env').write_text(
                '# comment\nRUNPOD_API_KEY="fixture-key"\n'
                'RUNPOD_ENDPOINT_ID=file-endpoint\nHF_TOKEN=must-not-load\n'
                'CLAIM_ENDPOINT_PAUSED=1\n', encoding='utf-8')
            with patch.dict(os.environ, {'RUNPOD_ENDPOINT_ID': 'process-endpoint', 'RUNPOD_API_KEY': 'stale-inherited-key'}, clear=True):
                client_config.load_local_env(directory)
                self.assertEqual(os.environ['RUNPOD_ENDPOINT_ID'], 'file-endpoint')
                self.assertEqual(os.environ['RUNPOD_API_KEY'], 'fixture-key')
                self.assertEqual(os.environ['CLAIM_ENDPOINT_PAUSED'], '1')
                self.assertNotIn('HF_TOKEN', os.environ)

    def test_check_is_local_and_redacts_key(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {'RUNPOD_API_KEY': 'fixture-secret',
                                    'RUNPOD_ENDPOINT_ID': 'custom-endpoint'}, clear=True), \
             patch.object(launch_test, 'load_local_env'), \
             patch.object(sys, 'argv', ['launch_test.py', '--check']), \
             patch.object(claim_client.urllib.request, 'urlopen',
                          side_effect=AssertionError('Network forbidden')), \
             contextlib.redirect_stdout(stream):
            launch_test.main()
        report = json.loads(stream.getvalue())
        self.assertEqual(report['endpoint'], 'custom-endpoint')
        self.assertEqual(report['model'], 'claim-v3')
        self.assertTrue(report['api_key_configured'])
        self.assertFalse(report['remote_status_checked'])
        self.assertNotIn('fixture-secret', stream.getvalue())

    def test_missing_or_blank_file_uses_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'RUNPOD_API_KEY': 'environment-key'}, clear=True):
                client_config.load_local_env(directory)
                self.assertEqual(os.environ['RUNPOD_API_KEY'], 'environment-key')
                Path(directory, '.env').write_text('RUNPOD_API_KEY=\n', encoding='utf-8')
                client_config.load_local_env(directory)
                self.assertEqual(os.environ['RUNPOD_API_KEY'], 'environment-key')

    def test_pause_blocks_cli_and_python_api_before_submission(self):
        with patch.dict(os.environ, {'CLAIM_ENDPOINT_PAUSED': '1'}), \
             patch.object(claim_client, 'post') as submit:
            with self.assertRaisesRegex(RuntimeError, 'paused'):
                next(claim_client.iter_job('endpoint', 'key', [], 256, 0))
            submit.assert_not_called()

    def test_default_python_api_routes_v3(self):
        with patch.dict(os.environ, {'CLAIM_ENDPOINT_PAUSED': '0'}), \
             patch.object(claim_client, 'post', return_value={'id': 'fixture-job'}) as submit:
            generator = claim_client.iter_job('endpoint', 'key', [], 256, 0)
            next(generator)
            self.assertEqual(submit.call_args.args[1]['input']['openai_input']['model'], 'claim-v3')
            generator.close()
            self.assertTrue(submit.call_args.args[0].endswith('/cancel/fixture-job'))

    def test_config_copies_match(self):
        self.assertEqual((ROOT / 'serving/client_config.py').read_bytes(),
                         (ROOT / 'space/client_config.py').read_bytes())


if __name__ == '__main__':
    unittest.main()

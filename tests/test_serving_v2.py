"""Offline contracts for adapter routing, cancellation, and the test UI."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
import claim_client as client
sys.path.insert(0, str(ROOT / "space"))
import test_ui as ui


def completion(model="claim-v2"):
    return {"id": "job-test", "status": "COMPLETED", "output": {
        "model": model, "choices": [{"message": {"content": "하우징을 포함하는 장치."},
                                      "finish_reason": "stop"}]}}


class ServingTests(unittest.TestCase):
    def test_paused_ui_never_submits_paid_request(self):
        with patch.object(ui, 'IS_PAUSED', True), patch.object(ui, 'iter_job') as invoke:
            with self.assertRaises(Exception):
                next(ui.generate([], '', 'ko', 'claim-v2', False, False, 0, 512))
            invoke.assert_not_called()

    def test_passthrough_routes_adapter_and_token_limit(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(completion()).encode()
        with patch.object(client, "post", return_value={"id": "job-test", "status": "IN_QUEUE"}) as post, \
             patch.object(client.urllib.request, "urlopen", return_value=response), \
             patch.object(client.time, "sleep"):
            events = list(client.iter_job("endpoint", "secret-test", [{"role": "user", "content": "drawings"}], 1024, 0, "claim-v2"))
        body = post.call_args.args[1]
        request = body["input"]["openai_input"]
        self.assertEqual(request["model"], "claim-v2")
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(body["input"]["openai_route"], "/v1/chat/completions")
        self.assertEqual(events[-1]["status"], "COMPLETED")
        self.assertNotIn("secret-test", json.dumps(body))

    def test_timeout_cancels_only_submitted_job(self):
        with patch.object(client, "post", return_value={"id": "job-test", "status": "IN_QUEUE"}) as post:
            with self.assertRaises(TimeoutError):
                list(client.iter_job("endpoint", "key", [], 10, 0, timeout=0))
        self.assertTrue(post.call_args.args[0].endswith("/cancel/job-test"))

    def test_rejects_base_fallback(self):
        with self.assertRaises(RuntimeError):
            client.response_metadata(completion("gemma4-31b"), "claim-v2")

    def test_closing_generator_cancels_pending_job(self):
        with patch.object(client, "post", return_value={"id": "job-test", "status": "IN_QUEUE"}) as post:
            job = client.iter_job("endpoint", "key", [], 10, 0)
            next(job)
            job.close()
        self.assertTrue(post.call_args.args[0].endswith("/cancel/job-test"))

    def test_ui_comparison_has_same_drawings_and_correct_prompts(self):
        calls = []
        def fake_job(endpoint, key, messages, tokens, temperature, model):
            calls.append((messages, model))
            yield completion(model)
        with patch.object(ui, "API_KEY", "test"), patch.object(ui, "IS_RL", False), \
             patch.object(ui, "TUNED_MODEL", "claim-v2"), patch.object(client, "TUNED_MODEL", "claim-v2"), \
             patch.object(ui.os.path, "getsize", return_value=10), \
             patch.object(client, "encode_image", side_effect=lambda p: "data:image/png;base64," + p), \
             patch.object(ui, "iter_job", side_effect=fake_job):
            results = list(ui.generate(["first", "second"], "", "ko", "claim-v2", False, True, 0, 1024))
        self.assertEqual([c[1] for c in calls], ["claim-v2", "gemma4-31b"])
        self.assertEqual(len(calls[0][0]), 1)
        self.assertEqual(calls[1][0][0]["role"], "system")
        self.assertEqual(calls[0][0][-1], calls[1][0][-1])
        content = calls[0][0][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].endswith("first"))
        self.assertTrue(content[1]["image_url"]["url"].endswith("second"))
        self.assertEqual(content[-1]["text"], client.TRAINING_PROMPTS["ko"])
        self.assertEqual(len(results[-1][-1]), 2)

    def test_rl_ui_keeps_json_and_submits_only_selected_adapter(self):
        calls = []
        raw = '```json\n{"claims":[],"annotations":[],"abstentions":["Example"]}\n```'
        def fake_job(endpoint, key, messages, tokens, temperature, model):
            calls.append((messages, model))
            value = completion(model)
            value['output']['choices'][0]['message']['content'] = raw
            yield value
        with patch.object(ui, 'API_KEY', 'test'), patch.object(ui, 'IS_RL', True), \
             patch.object(client, 'TUNED_MODEL', 'claim-v3'), \
             patch.object(ui.os.path, 'getsize', return_value=10), \
             patch.object(client, 'encode_image', return_value='data:image/png;base64,test'), \
             patch.object(ui, 'iter_job', side_effect=fake_job):
            results = list(ui.generate(['drawing'], '', 'ko', 'claim-v3', False, True, 0, 1024))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 'claim-v3')
        self.assertEqual(calls[0][0][0]['content'][-1]['text'], client.RL_PROMPTS['ko'])
        self.assertEqual(results[-1][0], raw)
        self.assertEqual(json.loads(results[-1][1])['abstentions'], ['Example'])
        self.assertTrue(results[-1][-1][0]['json_valid'])

    def test_deployed_client_is_in_sync(self):
        self.assertEqual((ROOT / "serving/claim_client.py").read_bytes(),
                         (ROOT / "space/claim_client.py").read_bytes())


if __name__ == "__main__":
    unittest.main()

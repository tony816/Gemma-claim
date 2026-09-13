"""Meaningful negative tests for policy leakage and semantic reward caps."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rl_judge import DIMENSIONS, RELEASE, rows, score
from tools.rl_preflight import policy_messages
from serving.rl_client import policies, evidence_in_policy, resolve_response
import json


class ExecutionGates(unittest.TestCase):
    def test_judge_control_character_transport_preserves_content_only(self):
        from rl_training.judge_transport import parse_judge_json
        from rl_training.core import parse_json
        raw = '{"reason":"literal\tcontrol", "score":1}'
        parsed, normalized = parse_judge_json(raw)
        self.assertTrue(normalized)
        self.assertEqual(parsed, {'reason':'literal\tcontrol', 'score':1})
        with self.assertRaises(json.JSONDecodeError):
            parse_json(raw)  # Policy output validation is not loosened.
        with self.assertRaises(json.JSONDecodeError):
            parse_judge_json('{"score":1')

    def test_transient_missing_connection_does_not_delete_established_pod(self):
        from rl_training.guard import missing_connection_action
        record={'started_at':100,'deadline':10000}
        self.assertEqual(missing_connection_action(3279,record,True),'retry')
        self.assertEqual(missing_connection_action(10000,record,True),'stop_at_deadline')
        self.assertEqual(missing_connection_action(500,record,False),'retry')
        self.assertEqual(missing_connection_action(1100,record,False),'delete_unprovisioned')

    def policy(self):
        return {'episode_id': 'example', 'task_type': 'claim_set',
                'images': ['first.png', 'second.png'],
                'messages': [{'role': 'user', 'content': 'Drawings only'}]}

    def candidate(self):
        return {'dimensions': dict(DIMENSIONS['claim_set']),
                'format_valid': True, 'severe_error': False,
                'severe_reasons': [], 'reasons': ['Image 1: parallel channels.']}

    def test_reward_fields_cannot_cross_policy_boundary(self):
        for field in ['reference', 'chosen', 'rejected', 'required_points', 'review']:
            row = self.policy(); row[field] = 'oracle'
            with self.assertRaises(ValueError):
                policy_messages(row)

    def test_assistant_answer_cannot_cross_policy_boundary(self):
        row = self.policy(); row['messages'].append({'role': 'assistant', 'content': 'answer'})
        with self.assertRaises(ValueError):
            policy_messages(row)

    def test_image_count_and_input_are_preserved_without_mutation(self):
        row = self.policy(); before = copy.deepcopy(row)
        messages = policy_messages(row)
        self.assertEqual(row, before)
        self.assertEqual([x['type'] for x in messages[0]['content']], ['image', 'image', 'text'])

    def test_high_dimension_scores_do_not_override_severe_cap(self):
        value = self.candidate(); value.update(severe_error=True, severe_reasons=['Unsupported valve'])
        self.assertEqual(score(value, 'claim_set'), .25)

    def test_unusable_format_receives_no_positive_reward(self):
        value = self.candidate(); value['format_valid'] = False
        self.assertEqual(score(value, 'claim_set'), 0)

    def test_judge_schema_fails_closed(self):
        value = self.candidate(); value['dimensions']['image_support'] = 10
        with self.assertRaises(ValueError): score(value, 'claim_set')
        value = self.candidate(); value['dimensions']['image_support'] = True
        with self.assertRaises(ValueError): score(value, 'claim_set')

    def test_judge_must_give_reasons_for_severe_error(self):
        value = self.candidate(); value['severe_error'] = True
        with self.assertRaises(ValueError): score(value, 'claim_set')

    def test_missing_dimensions_not_inferred(self):
        value = self.candidate(); del value['dimensions']['image_support']
        with self.assertRaises(ValueError): score(value, 'claim_set')

    def test_case_annotation_resolves_from_verified_card_store(self):
        inputs = policies()
        policy = next(p for p in inputs.values() if evidence_in_policy(p)[0])
        reference = next(r for r in rows(RELEASE/'reward/references.train.jsonl')
                         if r['episode_id'] == policy['episode_id'])['response']
        _, resolved = resolve_response(json.dumps(reference), policy)
        self.assertTrue(resolved)
        self.assertEqual(len(resolved[0]['source']['source_sha256']), 64)
        self.assertTrue(resolved[0]['source']['docket'])
        self.assertTrue(resolved[0]['source']['pages'])

    def test_unsupplied_annotation_is_not_resolved(self):
        policy = next(p for p in policies().values() if evidence_in_policy(p)[0])
        reference = next(r for r in rows(RELEASE/'reward/references.train.jsonl')
                         if r['episode_id'] == policy['episode_id'])['response']
        reference['annotations'][0]['card_id'] = 'invented'
        with self.assertRaises(ValueError):
            resolve_response(json.dumps(reference), policy)

    def test_calibration_drawing_resolves_explicitly_supplied_train_card(self):
        policy = next(p for p in rows(RELEASE/'policy/calibration.jsonl')
                      if p['task_type'] == 'claim_set' and evidence_in_policy(p)[0])
        supplied = {c['card_id'] for c in evidence_in_policy(policy)[0]}
        train_cards = {c['card_id'] for c in rows(RELEASE/'evidence/cards.train.jsonl')}
        self.assertTrue(supplied.issubset(train_cards))
        reference = next(r for r in rows(RELEASE/'reward/references.calibration.jsonl')
                         if r['episode_id'] == policy['episode_id'])['response']
        _, resolved = resolve_response(json.dumps(reference), policy, source_split='calibration')
        self.assertTrue(resolved)
        self.assertEqual(len(resolved[0]['source']['source_sha256']), 64)
        reference['annotations'][0]['card_id'] = next(iter(train_cards-supplied))
        with self.assertRaises(ValueError):
            resolve_response(json.dumps(reference), policy, source_split='calibration')

    def test_unsupplied_advisory_citation_is_not_resolved(self):
        policy = next(p for p in policies().values() if p['task_type'] == 'patent_advisory')
        with self.assertRaises(ValueError):
            resolve_response(json.dumps({'answer': 'An answer', 'citations': ['invented'],
                                         'limitations': []}), policy)

    def test_ui_exposes_only_train_examples(self):
        visible = set(policies())
        self.assertEqual(len(visible), 88)
        for split in ('calibration', 'test'):
            self.assertTrue(visible.isdisjoint(r['episode_id'] for r in rows(RELEASE/f'policy/{split}.jsonl')))

    def test_display_fence_unwrapping_does_not_change_strict_evaluation(self):
        policy = next(p for p in policies().values() if p['task_type'] == 'patent_advisory')
        raw = '```json\n' + json.dumps({'answer': 'Example', 'citations': [], 'limitations': []}) + '\n```'
        with self.assertRaises(ValueError): resolve_response(raw, policy)
        parsed, _ = resolve_response(raw, policy, allow_code_fence=True)
        self.assertEqual(parsed['answer'], 'Example')


if __name__ == '__main__':
    unittest.main()

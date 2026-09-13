import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import json
import hashlib

from tools.build_rl_lineage import Graph, publication_key, application_key, case_lookup_ids
from tools.validate_rl_curation import canonical_hash
from tools.calibrate_rl_material_judge import scored_candidate
from tools.assemble_rl_release import approved,annotation_errors
from tools.audit_rl_material_release import manifest_errors,policy_errors,calibration_errors
from tools.freeze_rl_material_release import review_gate_errors


class LineageBoundaryTests(unittest.TestCase):
    def test_publication_kind_and_reissue_identity(self):
        self.assertEqual(publication_key("US 5,658,261 A"), publication_key("US5658261"))
        self.assertNotEqual(publication_key("USRE33054"), publication_key("US33054"))

    def test_application_and_publication_namespaces_do_not_collide(self):
        self.assertNotEqual(application_key("US19/053,195"), publication_key("US19053195"))
        self.assertEqual(application_key("US202163217680P"), application_key("US63/217,680"))

    def test_family_overlap_is_transitive(self):
        graph = Graph()
        graph.join(["pub:a", "app:parent"])
        graph.join(["pub:b", "app:parent"])
        graph.join(["pub:b", "pub:c"])
        self.assertTrue(graph.same("pub:a", "pub:c"))
        self.assertFalse(graph.same("pub:a", "pub:d"))

    def test_case_identifier_type_is_not_lost(self):
        row = {"case_id":"test", "case_patent_ids":[{"id":"399364", "provenance":{"quote":"실용신안권자"}}]}
        self.assertEqual(case_lookup_ids(row, {}), ["KR200399364"])

    def test_semantic_change_invalidates_hash(self):
        self.assertNotEqual(canonical_hash({"answer":"supported"}), canonical_hash({"answer":"unsupported"}))
        self.assertEqual(canonical_hash({"a":1,"b":2}), canonical_hash({"b":2,"a":1}))

    def test_reward_caps_severe_error_and_rejects_invalid_dimensions(self):
        scores={'image_support':4,'dependency_scope':3,'source_annotation':2,'uncertainty':1}
        self.assertEqual(scored_candidate({'dimensions':scores,'severe_error':False},'claim_set'),1)
        self.assertEqual(scored_candidate({'dimensions':scores,'severe_error':True},'claim_set'),.25)
        self.assertIsNone(scored_candidate({'dimensions':dict(scores,image_support=5),'severe_error':False},'claim_set'))
        self.assertIsNone(scored_candidate({'dimensions':dict(scores,image_support=True),'severe_error':False},'claim_set'))


class ReleaseFailureTests(unittest.TestCase):
    def test_asset_corruption_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);(base/'image.png').write_bytes(b'original')
            files={'image.png':hashlib.sha256(b'original').hexdigest()}
            manifest={'files':files,'content_digest':canonical_hash(files)}
            self.assertEqual(manifest_errors(base,manifest),[])
            (base/'extra.txt').write_text('unmanifested')
            self.assertIn('manifest_unlisted_member:extra.txt',manifest_errors(base,manifest))
            (base/'image.png').write_bytes(b'changed')
            self.assertIn('manifest_member:image.png',manifest_errors(base,manifest))
            files={'../escape':'invalid'}
            self.assertIn('manifest_path_escape:../escape',manifest_errors(base,{'files':files,'content_digest':canonical_hash(files)}))

    def test_semantic_review_requires_current_hash_and_independent_reviewer(self):
        row={'record_id':'x','author':'author','answer':'a'};digest=canonical_hash(row)
        review={'independent_from_author':True,'reviewer':'reviewer','author_agent':'author',
            'records':[{'record_id':'x','source_record_sha256':digest,'verdict':'accept','evidence_checked':['page1']}]}
        source={'record_hashes':[{'record_id':'x','sha256':digest,'errors':[]}]}
        with patch('tools.assemble_rl_release.read_json',side_effect=lambda p,d: review if 'semantic' in str(p) else source):
            self.assertTrue(approved(row,'test')[0])
            self.assertFalse(approved({**row,'answer':'changed'},'test')[0])
            review['reviewer']='author'
            self.assertFalse(approved(row,'test')[0])

    def test_reference_and_assistant_messages_never_enter_policy(self):
        ref={'response':{'answer':'sensitive reward answer'}}
        p={'episode_id':'x','task_type':'patent_advisory','images':[],'messages':[{'role':'user','content':'question'}]}
        self.assertEqual(policy_errors(p,ref),[])
        p['messages'].append({'role':'assistant','content':'sensitive reward answer'})
        issues=policy_errors(p,ref)
        self.assertIn('policy_message_schema_or_assistant_leak',issues)
        self.assertIn('reference_answer_leak',issues)
        p={'episode_id':'short','task_type':'claim_set','images':[],
            'messages':[{'role':'user','content':'Gold example: A tube.'}]}
        self.assertIn('reference_claim_leak',policy_errors(p,{'response':{'claims':[{'text':'A tube.'}]}}))

    def test_annotation_rejects_target_family_and_missing_or_wrong_case(self):
        drawing={'record_id':'d','source_split':'train','claims':['A housing.'],'parents':[[]],'abstentions':[]}
        card={'case_id':'c','jurisdiction':'US','review_status':'approved','lineage_key':'case-family','review':{'record_sha256':'casehash'}}
        a={'claim_number':1,'card_id':'p','mode':'provided_during_drafting','application':'Specific application'}
        ext={'drawing_record_id':'d','drawing_record_sha256':canonical_hash(drawing),'source_split':'train',
            'jurisdiction':'US','card_ids':['p'],'case_record_sha256':{'c':'casehash'},'annotations':[a],
            'inferior_annotations':[{**a,'application':'Wrong application'}],'required_points':['point'],
            'forbidden_claims':['error'],'preference_reason':'substantive error'}
        self.assertEqual(annotation_errors(ext,drawing,{'p':card},'train','target-family'),[])
        self.assertIn('annotation_target_family',annotation_errors(ext,drawing,{'p':card},'train','case-family'))
        self.assertIn('annotation_target_lineage_missing',annotation_errors(ext,drawing,{'p':card},'train',None))
        self.assertIn('annotation_supplied_card_ids',annotation_errors(ext,drawing,{},'train','target-family'))
        ext['case_record_sha256']['c']='stale'
        self.assertIn('annotation_case_binding',annotation_errors(ext,drawing,{'p':card},'train','target-family'))

    def test_missing_calibration_and_empty_gold_never_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            errors=calibration_errors(Path(temp),[{'pair_id':'x'}])
        self.assertIn('blind_calibration_not_passed',errors)
        self.assertIn('calibration_pairs_stale',errors)

    def test_freeze_requires_current_independent_code_and_qa(self):
        required=('assemble_rl_release.py','audit_rl_material_release.py','freeze_rl_material_release.py',
            'validate_rl_annotations.py','build_rl_lineage.py','rl_family_metadata.py',
            'calibrate_rl_material_judge.py','probe_rl_material_release.py','validate_rl_curation.py')
        files={'tools/'+n:hashlib.sha256((Path('tools')/n).read_bytes()).hexdigest() for n in required}
        for name in ('tests/test_rl_material_release.py','tests/test_rl_family_metadata.py',
                     'rl_materials/MATERIALS_SPEC.md','rl_materials/REWARD_PROTOCOL.md'):
            files[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        review={'recommendation':'APPROVE','codeQualityStatus':'CLEAR','reviewer':'independent-code',
                'independent_from_author':True,'reviewed_files':files}
        qa={'passed':True,'reviewer':'independent-qa','independent_from_author':True,'candidate_content_digest':'d',
            'scenarios':[{'passed':True,'evidence':'artifact'} for _ in range(3)]}
        with patch('tools.freeze_rl_material_release.read_json',side_effect=lambda p,d: review if 'code-review' in str(p) else qa):
            self.assertEqual(review_gate_errors({'content_digest':'d'}),[])
            before=files.pop('rl_materials/REWARD_PROTOCOL.md')
            self.assertIn('code_review_stale:rl_materials/REWARD_PROTOCOL.md',review_gate_errors({'content_digest':'d'}))
            files['rl_materials/REWARD_PROTOCOL.md']=before
            qa['candidate_content_digest']='stale'
            self.assertIn('independent_release_qa_missing_or_stale',review_gate_errors({'content_digest':'d'}))
            files['tools/assemble_rl_release.py']='stale'
            self.assertIn('code_review_stale:tools/assemble_rl_release.py',review_gate_errors({'content_digest':'d'}))


if __name__ == "__main__":
    unittest.main()

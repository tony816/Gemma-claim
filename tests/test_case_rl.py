import copy
import unittest

from rl_materials.contracts import assess, eligible_cards, resolve_annotations, validate_output


class ClaimSetTests(unittest.TestCase):
    def setUp(self):
        self.cards = {"c1": {"review_status": "approved", "jurisdiction": "KR",
                      "docket": "example", "court": "fixture", "decision_date": "2000-01-01",
                      "pdf_path": "fixture.pdf", "source_sha256": "0" * 64, "pages": [1]}}
        self.output = {"claims": [
            {"number": 1, "kind": "independent", "depends_on": [], "text": "하우징을 포함하는 장치."},
            {"number": 2, "kind": "dependent", "depends_on": [1], "text": "제1항에 있어서, 상기 하우징은 개구를 포함하는 장치."}],
            "annotations": [{"claim_number": 2, "card_id": "c1", "mode": "provided_during_drafting", "application": "구조 관계 확인"}],
            "abstentions": []}

    def test_resolution_does_not_claim_causality(self):
        self.assertEqual(validate_output(self.output, self.cards), [])
        resolved = resolve_annotations(self.output, self.cards)[0]
        self.assertEqual(resolved["source"]["docket"], "example")
        self.assertEqual(resolved["mode"], "provided_during_drafting")

    def test_structural_success_is_not_legal_reward(self):
        report = assess(self.output, self.cards)
        self.assertTrue(report["structurally_valid"])
        self.assertIsNone(report["reward"])
        self.assertFalse(report["training_eligible"])

    def test_future_self_and_boolean_dependencies(self):
        for parent in [2, 3, True, 0]:
            with self.subTest(parent=parent):
                value = copy.deepcopy(self.output)
                value["claims"][1]["depends_on"] = [parent]
                self.assertIn("claim_2_dependency", validate_output(value, self.cards))

    def test_dependency_in_text_must_match(self):
        value = copy.deepcopy(self.output)
        value["claims"][1]["text"] = "제3항에 있어서, 개구가 있는 장치."
        self.assertIn("claim_2_text_dependency", validate_output(value, self.cards))

    def test_english_dependency(self):
        value = copy.deepcopy(self.output)
        value["claims"][1]["text"] = "The device of claim 1, wherein the housing has an opening."
        self.assertEqual(validate_output(value, self.cards), [])

    def test_invented_or_draft_sources_rejected(self):
        for cards in [{}, {"c1": {"review_status": "draft"}}]:
            self.assertIn("annotation_unapproved_or_unsupplied_card", validate_output(self.output, cards))
            with self.assertRaises(ValueError):
                resolve_annotations(self.output, cards)

    def test_no_cards_can_abstain_without_fake_citation(self):
        value = copy.deepcopy(self.output)
        value["annotations"] = []
        value["abstentions"] = ["판례 카드가 제공되지 않음"]
        self.assertEqual(validate_output(value, {}), [])

    def test_only_one_independent_and_bounded_count(self):
        value = copy.deepcopy(self.output)
        value["claims"][1]["kind"] = "independent"
        self.assertIn("claim_2_kind", validate_output(value, self.cards))
        self.assertIn("claim_count", validate_output(self.output, self.cards, max_dependents=0))

    def test_posthoc_cannot_be_misreported_as_drafting_source(self):
        self.output["annotations"][0]["mode"] = "posthoc_review"
        self.assertIn("annotation_mode", validate_output(self.output, self.cards))

    def test_malformed_inputs_do_not_crash(self):
        for value in [None, [], {}, {"claims": [None]}, {"claims": [{}]}]:
            self.assertTrue(validate_output(value, self.cards))
        self.output["annotations"][0]["card_id"] = []
        self.assertTrue(validate_output(self.output, self.cards))

    def test_retrieval_requires_jurisdiction_and_family_review(self):
        card = {**self.cards["c1"], "card_id": "c1", "source_text_verified": True,
                "generator_eligible": True, "legal_history_status": "reviewed",
                "case_family_status": "reviewed", "case_patent_family_ids": ["case-patent"]}
        self.assertEqual(list(eligible_cards([card], "KR", {"target-patent"})), ["c1"])
        self.assertEqual(eligible_cards([card], "US", {"target-patent"}), {})
        self.assertEqual(eligible_cards([card], "KR", {"case-patent"}), {})
        self.assertEqual(eligible_cards([card], "KR", set()), {})
        card["case_family_status"] = "unresolved"
        self.assertEqual(eligible_cards([card], "KR", {"target-patent"}), {})

    def test_duplicate_card_id_cannot_ambiguously_resolve(self):
        card = {**self.cards["c1"], "card_id": "c1", "source_text_verified": True,
                "generator_eligible": True, "legal_history_status": "reviewed",
                "case_family_status": "reviewed", "case_patent_family_ids": ["case-patent"]}
        with self.assertRaises(ValueError):
            eligible_cards([card, card], "KR", {"target"})


if __name__ == "__main__":
    unittest.main()

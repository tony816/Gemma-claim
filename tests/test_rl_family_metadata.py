import unittest

from tools.rl_family_metadata import metadata_from_html


class FamilyMetadataTests(unittest.TestCase):
    def test_prior_art_and_similar_documents_are_not_family(self):
        html = '''<section itemprop="family"><h2>ID=1234</h2>
        <tr itemprop="countryStatus"><td><a href="/patent/EP123A1/en">family</a></td></tr>
        <tr itemprop="backwardReferencesFamily"><td><a href="/patent/US555A1/en">prior art</a></td></tr>
        <tr itemprop="forwardReferencesFamily"><td><a href="/patent/US666B2/en">cites us</a></td></tr>
        <tr itemprop="similarDocuments"><td><a href="/patent/US777B2/en">similar</a></td></tr>
        </section>'''
        row = metadata_from_html("US123B2", html)
        self.assertEqual(row["related_publications"], ["EP123A1", "US123B2"])
        self.assertEqual(row["family_id"], "1234")

    def test_real_priority_and_parent_property_names_are_retained(self):
        html = '''<h2>ID=2345</h2>
        <tr itemprop="appsClaimingPriority"><td><span itemprop="applicationNumber">US63/761,538</span></td></tr>
        <tr itemprop="parentApps"><td><a href="/patent/US123A1/en">parent</a></td></tr>
        <tr itemprop="priorityApps"><td><a href="/patent/WO123A1/en">priority</a></td></tr>
        <tr itemprop="beforeApplications"><td><a href="/patent/US122B2/en">earlier</a></td></tr>
        <tr itemprop="pubs"><td><a href="/patent/US124A1/en">publication</a></td></tr>'''
        row = metadata_from_html("US124B2", html)
        self.assertEqual(row["application_numbers"], ["US63/761,538"])
        self.assertEqual(set(row["related_publications"]), {"US124B2", "US124A1", "US122B2", "US123A1", "WO123A1"})

    def test_missing_family_is_unresolved(self):
        self.assertEqual(metadata_from_html("US123B2", "<html></html>")["status"], "family_id_missing")


if __name__ == "__main__":
    unittest.main()

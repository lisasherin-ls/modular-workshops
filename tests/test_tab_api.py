import os
import unittest


class TabApiTest(unittest.TestCase):
    def test_get_race_rejects_missing_fixture_instead_of_using_warwick_farm(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from agents.tools_live import get_race

        previous = os.environ.get("TAB_FIXTURES")
        os.environ["TAB_FIXTURES"] = "1"
        try:
            with self.assertRaisesRegex(ValueError, r"no saved response for LIS R4.*saved file holds WFM R2"):
                get_race.invoke(
                    {
                        "date": "2026-09-23",
                        "venue_code": "LIS",
                        "race_number": 4,
                    }
                )
        finally:
            if previous is None:
                os.environ.pop("TAB_FIXTURES", None)
            else:
                os.environ["TAB_FIXTURES"] = previous

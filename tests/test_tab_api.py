import unittest

from agents.tab_api import going_from_track_condition


class GoingFromTrackConditionTests(unittest.TestCase):
    def test_known_track_conditions(self):
        expected = {
            "FIRM": "Firm",
            "FAST": "Firm",
            "GOOD": "Good",
            "GOOD3": "Good To Firm",
            "GOOD4": "Good",
            "AWT": "Good",
            "SOFT5": "Good To Soft",
            "SOFT6": "Soft",
            "SOFT7": "Soft",
            "DEAD": "Soft",
            "HVY8": "Heavy",
            "HVY9": "Heavy",
            "HEAVY": "Heavy",
        }

        for condition, going in expected.items():
            with self.subTest(condition=condition):
                self.assertEqual(going_from_track_condition(condition), going)

    def test_empty_or_unknown_track_conditions(self):
        self.assertIsNone(going_from_track_condition(None))
        self.assertIsNone(going_from_track_condition(""))
        with self.assertLogs("agents.tab_api", level="WARNING"):
            self.assertIsNone(going_from_track_condition("MUDDY"))


if __name__ == "__main__":
    unittest.main()

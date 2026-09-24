import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tab_api import normalize_meetings, resolve_venue
from agents.tools_live import list_meetings


def meeting(name, location="AU", code="", races=None):
    return {
        "meetingName": name,
        "location": location,
        "raceType": "R",
        "meetingDate": "2026-09-24",
        "venueMnemonic": code,
        "races": races if races is not None else [{"raceNumber": 1, "raceStatus": "Open"}],
    }


class VenueScopingTests(unittest.TestCase):
    def test_unaddressable_venue_requires_state_lookup(self):
        with self.assertRaisesRegex(ValueError, "listed but not addressable.*state-scoped"):
            resolve_venue({"meetings": [meeting("Mystery Track")]}, "Mystery")

    def test_ambiguous_venue_lists_unique_candidates(self):
        payload = {
            "meetings": [
                meeting("Belmont Park (USA)", "USA", "BEL"),
                meeting("Belmont Park", "AUS", "BEP"),
            ]
        }
        with self.assertRaisesRegex(ValueError, r"BEL - Belmont Park \(USA\).*BEP - Belmont Park \(AUS\)"):
            resolve_venue(payload, "Belmont")

    def test_venue_with_only_closed_races_uses_upcoming_error(self):
        payload = {
            "meetings": [
                meeting("Warwick Farm", "NSW", "WFM", [{"raceNumber": 1, "raceStatus": "Closed"}])
            ]
        }
        with patch("agents.tools_live.fetch_meetings", return_value=payload):
            with self.assertRaisesRegex(ValueError, "No thoroughbred meetings still to run"):
                list_meetings.func("2026-09-24", "NSW", "Warwick")

    def test_au_venue_returns_addressable_code(self):
        payload = {"meetings": [meeting("Warwick Farm", "NSW", "WFM")]}
        with patch("agents.tools_live.fetch_meetings", return_value=payload):
            result = list_meetings.func("2026-09-24", "ALL", "Warwick")
        self.assertEqual(result[0]["venue_code"], "WFM")

    def test_normalized_meeting_marks_going_and_card_completeness(self):
        payload = {
            "meetings": [
                {
                    **meeting(
                        "Warwick Farm",
                        "NSW",
                        "WFM",
                        [{"raceNumber": 1, "raceStatus": "Open"}, {"raceNumber": 3, "raceStatus": "Open"}],
                    ),
                    "trackCondition": None,
                }
            ]
        }
        result = normalize_meetings(payload)[0]
        self.assertFalse(result["going_available"])
        self.assertFalse(result["card_complete"])


if __name__ == "__main__":
    unittest.main()

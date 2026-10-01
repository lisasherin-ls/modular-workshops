import unittest

from agents.tab_api import normalize_meetings, normalize_race


def race_payload(location: str, timestamp: str = "2026-09-29T02:11:35Z") -> dict:
    return {
        "raceNumber": 1,
        "raceName": "Test Race",
        "raceStartTime": timestamp,
        "fixedOddsUpdateTime": timestamp,
        "meeting": {"location": location, "meetingName": "Test Meeting"},
        "runners": [],
    }


class TimestampNormalizationTests(unittest.TestCase):
    def test_nsw_timestamps_include_aest_and_date(self):
        normalized = normalize_race(race_payload("NSW"))

        self.assertEqual(normalized["odds_as_at_utc"], "2026-09-29T02:11:35Z")
        self.assertEqual(normalized["odds_as_at_local"], "12:11pm AEST, 29 September")
        self.assertEqual(normalized["race"]["start_time_utc"], "2026-09-29T02:11:35Z")
        self.assertEqual(normalized["race"]["start_time_local"], "12:11pm AEST, 29 September")

    def test_irl_timestamps_use_local_zone(self):
        normalized = normalize_race(race_payload("IRL"))

        self.assertEqual(normalized["odds_as_at_local"], "3:11am IST, 29 September")

    def test_unknown_location_labels_timestamps_as_utc(self):
        normalized = normalize_race(race_payload("UNKNOWN"))

        self.assertEqual(normalized["odds_as_at_local"], "2:11am UTC, 29 September")

    def test_meeting_starts_use_renamed_timestamp_fields(self):
        normalized = normalize_meetings(
            {
                "meetings": [
                    {
                        "raceType": "R",
                        "location": "NSW",
                        "races": [{"raceNumber": 1, "raceStartTime": "2026-09-29T02:11:35Z"}],
                    }
                ]
            }
        )

        race = normalized[0]["races"][0]
        self.assertEqual(race["start_time_local"], "12:11pm AEST, 29 September")
        self.assertNotIn("start_time", race)

    def test_rendered_timestamp_fields_are_not_zone_less(self):
        normalized = normalize_race(race_payload("NSW"))

        rendered = [
            normalized["odds_as_at_local"],
            normalized["race"]["start_time_local"],
        ]
        self.assertTrue(all(" " in value and ", " in value for value in rendered))


if __name__ == "__main__":
    unittest.main()

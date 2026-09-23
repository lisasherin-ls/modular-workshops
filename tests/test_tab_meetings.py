import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents import tools_live
from agents.tab_api import normalize_meetings


FIXTURE = Path(__file__).parents[1] / "evals" / "fixtures" / "tab_meetings_by_jurisdiction_info.json"


@pytest.fixture
def meetings_payload():
    return json.loads(FIXTURE.read_text())


def test_compact_all_meetings_stays_under_20k(meetings_payload):
    meetings = normalize_meetings(meetings_payload, state=None)

    assert len(json.dumps(meetings)) < 20_000


def test_venue_scoped_list_returns_one_meeting(monkeypatch, meetings_payload):
    monkeypatch.setattr(tools_live, "fetch_meetings", lambda date: meetings_payload)

    meetings = tools_live.list_meetings.invoke(
        {"date": "2026-09-23", "state": "ALL", "venue_code": "WFM"}
    )

    assert len(meetings) == 1
    assert meetings[0]["venue_code"] == "WFM"


def test_verbose_meetings_retain_full_race_shape(meetings_payload):
    meetings = normalize_meetings(meetings_payload, state="NSW", verbose=True)
    race = meetings[0]["races"][0]

    assert set(race) == {
        "race_number",
        "name",
        "distance_m",
        "start_time",
        "status",
        "scratched",
    }

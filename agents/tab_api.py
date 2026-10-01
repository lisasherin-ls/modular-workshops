"""Normalisation layer over the TAB info service.

The API answers broadly and expects the caller to narrow: a single meetings
request returns 66 meetings across 18 jurisdictions and three race types, and
a single race weighs 60KB, most of it betting plumbing. These functions cut
both down to what a tipster actually reasons over, and derive the handful of
fields that do the real work (implied probability, market move, wet form).

`jurisdiction` is the punter's betting jurisdiction, NOT a location filter —
filtering by state and race type happens here, client-side, the same way the
TAB web app does it.

The normalise functions are pure: hand them a parsed payload. `fetch_*` is a
thin wrapper for live calls, kept separate so everything below it can be
tested against the saved fixtures in evals/fixtures/.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

try:  # TAB's edge fingerprints the TLS handshake; plain clients are refused.
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - falls back to httpx
    curl_requests = None

from agents.mock_data import GOING_SCALE

# Overridable for a proxy or a mock server; falls back to the public endpoint.
# `or` rather than a getenv default so an empty value falls back too.
API_ROOT = os.getenv("API_ROOT") or "https://api.beta.tab.com.au/v1/tab-info-service/racing"
logger = logging.getLogger(__name__)

# Track condition comes back in at least four dialects across one response:
# the AU numeric scale (SOFT5), bare international words (GOOD), US usage
# (FAST), and synthetic tracks (AWT). 28 of 66 meetings carry none at all.
GOING_FROM_TRACK_CONDITION: dict[str, str] = {
    "FIRM1": "Firm", "FIRM2": "Firm", "FIRM": "Firm", "FAST": "Firm", "HARD": "Firm",
    "GOOD3": "Good To Firm",
    "GOOD4": "Good", "GOOD": "Good", "AWT": "Good", "SYNTHETIC": "Good", "STANDARD": "Good",
    "SOFT5": "Good To Soft",
    "SOFT6": "Soft", "SOFT7": "Soft", "SOFT": "Soft", "DEAD": "Soft", "SLOW": "Soft",
    "HEAVY8": "Heavy", "HEAVY9": "Heavy", "HEAVY10": "Heavy", "HEAVY": "Heavy",
    "HVY": "Heavy", "HVY8": "Heavy", "HVY9": "Heavy", "HVY10": "Heavy",
}

GOING_PREFIX_ALIASES = {
    "FIRM": "FIRM",
    "FAST": "FIRM",
    "HARD": "FIRM",
    "GOOD": "GOOD",
    "AWT": "GOOD",
    "SYNTHETIC": "GOOD",
    "STANDARD": "GOOD",
    "SOFT": "SOFT",
    "DEAD": "SOFT",
    "SLOW": "SOFT",
    "HEAVY": "HEAVY",
    "HVY": "HEAVY",
}

# A race in one of these states has been run. Never tip into it.
FINISHED_STATUSES = {"Paying", "Interim", "Closed", "Abandoned", "Final"}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VENUE_RE = re.compile(r"^[A-Za-z]{2,6}$")
_JURISDICTION_RE = re.compile(r"^[A-Za-z]{2,3}$")


def going_from_track_condition(condition: str | None) -> str | None:
    """Map a TAB track condition onto our going scale, or None if unknown."""
    normalized = (condition or "").strip().upper()
    if not normalized:
        return None
    match = re.fullmatch(r"([A-Z]+)(\d*)", normalized)
    if not match:
        logger.warning("Unknown TAB track condition: %s", condition)
        return None
    prefix, suffix = match.groups()
    canonical_prefix = GOING_PREFIX_ALIASES.get(prefix)
    going = GOING_FROM_TRACK_CONDITION.get(f"{canonical_prefix or prefix}{suffix}")
    if going not in GOING_SCALE:
        logger.warning("Unknown TAB track condition: %s", condition)
        return None
    return going


def _tcdw(indicators: str | None) -> dict[str, bool]:
    """Unpack the TCDW string: Track, Course, Distance, Wet.

    Uppercase means the horse has won under that condition, lowercase means
    it has placed. "C W" is a course winner and a wet-track winner.
    """
    ind = indicators or ""
    return {
        "track_winner": "T" in ind,
        "course_winner": "C" in ind,
        "distance_winner": "D" in ind,
        "wet_winner": "W" in ind,
        "wet_placed": "w" in ind,
    }


def normalize_meetings(
    payload: dict,
    state: str | None = "NSW",
    race_type: str = "R",
    upcoming_only: bool = True,
) -> list[dict]:
    """Today's meetings, filtered to what we care about.

    Args:
        payload: the parsed meetings response.
        state: keep only this location (e.g. "NSW"); None keeps every location.
        race_type: "R" thoroughbred, "G" greyhound, "H" harness.
        upcoming_only: drop races that have already been run.
    """
    meetings = []
    for meeting in payload.get("meetings") or []:
        if meeting.get("raceType") != race_type:
            continue
        if state and meeting.get("location") != state:
            continue

        races = []
        for race in meeting.get("races") or []:
            if upcoming_only and race.get("raceStatus") in FINISHED_STATUSES:
                continue
            races.append(
                {
                    "race_number": race.get("raceNumber"),
                    "name": race.get("raceName"),
                    "distance_m": race.get("raceDistance"),
                    "start_time": race.get("raceStartTime"),
                    "status": race.get("raceStatus"),
                    "scratched": [
                        s.get("runnerNumber") for s in (race.get("scratchings") or [])
                    ],
                }
            )
        if not races:
            continue

        condition = meeting.get("trackCondition")
        meetings.append(
            {
                "venue": meeting.get("meetingName"),
                "venue_code": meeting.get("venueMnemonic"),
                "state": meeting.get("location"),
                "date": meeting.get("meetingDate"),
                "track_condition": condition,
                "going": going_from_track_condition(condition),
                "weather": meeting.get("weatherCondition"),
                "rail": meeting.get("railPosition"),
                "races": races,
            }
        )
    return meetings


def _runner(runner: dict, market_total: float, include_prices: bool) -> dict:
    fixed = runner.get("fixedOdds") or {}
    win, opening = fixed.get("returnWin"), fixed.get("returnWinOpen")

    # claimAmount is -1 when the rider claims nothing.
    claim = runner.get("claimAmount") or 0
    weight = runner.get("handicapWeight")

    out = {
        "number": runner.get("runnerNumber"),
        "name": runner.get("runnerName"),
        "trainer": runner.get("trainerFullName") or runner.get("trainerName"),
        "jockey": runner.get("riderDriverFullName") or runner.get("riderDriverName"),
        "barrier": runner.get("barrierNumber"),
        "weight_kg": weight,
        "effective_weight_kg": round(weight - claim, 1) if weight and claim > 0 else weight,
        "last_5": runner.get("last5Starts"),
        "form_rating": runner.get("dfsFormRating"),
        "early_speed": runner.get("earlySpeedRatingBand"),
        "blinkers": runner.get("blinkers"),
        **_tcdw(runner.get("tcdwIndicators")),
    }

    if include_prices and win:
        out |= {
            "win_odds": win,
            "place_odds": fixed.get("returnPlace"),
            "opening_odds": opening,
            "market_move": (
                "firmed" if opening and win < opening
                else "drifted" if opening and win > opening
                else "steady"
            ),
            # What the market reckons its chance is, as a percentage of the
            # whole book. The agent's job is to disagree with this usefully.
            "implied_win_pct": round(100 / win / market_total, 1) if market_total else None,
            "favourite": bool(fixed.get("isFavouriteWin")),
        }
    return out


def _race_shape(runners: list[dict]) -> str:
    """One line on how the race is likely to be run."""
    leaders = [r["name"] for r in runners if r.get("early_speed") == "LEADER"]
    if len(leaders) >= 3:
        return f"{len(leaders)} genuine leaders ({', '.join(leaders)}) — expect a contested lead"
    if len(leaders) == 1:
        return f"{leaders[0]} looks the only leader — could get an easy time in front"
    return f"{len(leaders)} leaders — even tempo likely"


def normalize_race(payload: dict, include_prices: bool = True) -> dict:
    """One race, cut down to what the tipster reasons over.

    Scratched runners and any whose betting is not open are dropped before
    the agent ever sees them. Editorial (a named tipster's selections, TAB's
    rating lists, the preview) is kept as structured fields carrying their
    source, so the agent can cite rather than absorb them.
    """
    meeting = payload.get("meeting") or {}
    condition = meeting.get("trackCondition")

    live = [
        r
        for r in payload.get("runners") or []
        if not r.get("vacantBox")
        and (r.get("fixedOdds") or {}).get("bettingStatus", "Open") == "Open"
    ]

    # Normalise the book so implied percentages sum to ~100 across the field.
    market_total = sum(
        1 / (r.get("fixedOdds") or {}).get("returnWin", 0)
        for r in live
        if (r.get("fixedOdds") or {}).get("returnWin")
    )
    runners = [_runner(r, market_total, include_prices) for r in live]

    return {
        "race": {
            "number": payload.get("raceNumber"),
            "name": payload.get("raceName"),
            "distance_m": payload.get("raceDistance"),
            "class": payload.get("raceClassConditions"),
            "prize": payload.get("prizeMoney"),
            "start_time": payload.get("raceStartTime"),
            "direction": payload.get("trackDirection"),
            "status": payload.get("raceStatus"),
            "places_paid": payload.get("numberOfPlaces"),
        },
        "meeting": {
            "venue": meeting.get("meetingName"),
            "venue_code": meeting.get("venueMnemonic"),
            "state": meeting.get("location"),
            "track_condition": condition,
            "going": going_from_track_condition(condition),
            "weather": meeting.get("weatherCondition"),
        },
        # Prices move. Anything quoting one must say when it was true.
        "odds_as_at": payload.get("fixedOddsUpdateTime") if include_prices else None,
        "runners": runners,
        "race_shape": _race_shape(runners),
        "editorial": _editorial(payload),
    }


def _editorial(payload: dict) -> dict:
    """Third-party opinion, kept with its attribution attached."""
    tips = payload.get("tips") or {}
    comments = payload.get("raceComments")

    preview, shortlist = None, []
    if comments:
        try:
            # It arrives as JSON encoded inside a JSON string.
            inner = json.loads(comments)
            preview, shortlist = inner.get("comment"), inner.get("runners") or []
        except (TypeError, ValueError):
            preview = str(comments)

    return {
        "tipster": tips.get("tipster"),
        "tipster_selections": tips.get("tipRunnerNumbers") or [],
        "tab_ratings": {
            r.get("ratingType"): r.get("ratingRunnerNumbers")
            for r in payload.get("ratings") or []
        },
        "preview_shortlist": shortlist,
        "preview": preview,
    }


# The form endpoint buckets a horse's record by track condition using both the
# current vocabulary (firm/good/soft/heavy) and the legacy one (dead/slow).
# Map them onto the going names we already use everywhere else.
GOING_BUCKETS = {
    "firm": "Firm",
    "good": "Good",
    "dead": "Good To Soft",
    "slow": "Soft",
    "soft": "Soft",
    "heavy": "Heavy",
}


def _record(summary: dict | None) -> dict:
    """A {starts, wins, placings} block, plus the strike rate if there are starts."""
    s = summary or {}
    starts = s.get("numberOfStarts") or 0
    wins = s.get("numberOfWins") or 0
    return {
        "starts": starts,
        "wins": wins,
        "placings": s.get("numberOfPlacings") or 0,
        "win_pct": round(wins / starts * 100) if starts else None,
    }


def _start(start: dict) -> dict:
    """One previous run, without the video links."""
    return {
        "date": start.get("startDate"),
        "venue": start.get("venueAbbreviation"),
        "distance_m": start.get("distance"),
        "going": start.get("trackCondition"),
        "class": start.get("class"),
        "finished": f"{start.get('finishingPosition')} of {start.get('numberOfStarters')}",
        "margin": start.get("margin"),
        "odds": start.get("odds"),
        "rider": start.get("rider"),
    }


def normalize_form(payload: dict) -> dict:
    """One runner's form history, cut to what a tipster reads.

    The per-going record is the valuable part: real starts and wins on each
    surface, rather than the single wet/dry flag the race payload carries.

    Trials are kept apart from races on purpose — a trial placing is not form,
    and counting the two together flatters a horse coming back from a spell.
    """
    starts = payload.get("runnerStarts") or {}
    summaries = starts.get("startSummaries") or {}
    trainer = (payload.get("trainerStarts") or {}).get("startSummaries") or {}
    rider = (payload.get("riderDriverStarts") or {}).get("startSummaries") or {}

    previous = starts.get("previousStarts") or []
    races = [_start(s) for s in previous if s.get("startType") != "Trial"]
    trials = [_start(s) for s in previous if s.get("startType") == "Trial"]

    # Several API buckets can map to one of our goings (slow and soft both
    # mean Soft), so add them together rather than letting one overwrite.
    by_going: dict[str, dict] = {}
    for api_name, going in GOING_BUCKETS.items():
        rec = _record(summaries.get(api_name))
        if not rec["starts"]:
            continue
        prior = by_going.get(going)
        if prior:
            total = prior["starts"] + rec["starts"]
            wins = prior["wins"] + rec["wins"]
            by_going[going] = {
                "starts": total,
                "wins": wins,
                "placings": prior["placings"] + rec["placings"],
                "win_pct": round(wins / total * 100),
            }
        else:
            by_going[going] = rec

    return {
        "runner": {
            "number": payload.get("runnerNumber"),
            "name": payload.get("runnerName"),
            "age": payload.get("age"),
            "sex": payload.get("sex"),
            "sire": payload.get("sire"),
            "dam": payload.get("dam"),
            "trainer": payload.get("trainerName"),
            "trainer_base": payload.get("trainerLocation"),
            "jockey": payload.get("riderOrDriver"),
        },
        "record": {
            "overall": _record(summaries.get("overall")),
            "by_going": by_going,
            "at_track": _record(summaries.get("track")),
            "at_distance": _record(summaries.get("distance")),
            "at_track_and_distance": _record(summaries.get("trackDistance")),
            "same_class": _record(summaries.get("classSame")),
            "stronger_class": _record(summaries.get("classStronger")),
            "first_up": _record(summaries.get("firstUp")),
            "second_up": _record(summaries.get("secondUp")),
        },
        "context": {
            "days_since_last_run": payload.get("daysSinceLastRun"),
            "runs_since_spell": payload.get("runsSinceSpell"),
            "class_move": payload.get("classLevel"),
            "field_strength": payload.get("fieldStrength"),
            "last_20_starts": payload.get("last20Starts"),
            "career_prize": payload.get("prizeMoney"),
            "winning_distances_m": [
                w.get("winningDistance") for w in payload.get("winningDistances") or []
            ],
        },
        "races": races,
        "trials": trials,
        "trainer_strike": {
            "at_track": _record(trainer.get("track")),
            "last_12_months": _record(trainer.get("last12Months")),
            "with_this_jockey": _record(trainer.get("jockey")),
        },
        "jockey_strike": {
            "at_track": _record(rider.get("track")),
            "last_12_months": _record(rider.get("last12Months")),
            "on_this_runner": _record(rider.get("runner")),
        },
        "comment": payload.get("formComment"),
    }


def _check(date: str, venue: str, jurisdiction: str) -> None:
    """Validate anything that goes into a URL, so a tool argument can't redirect it."""
    if not _DATE_RE.match(date):
        raise ValueError(f"date must be YYYY-MM-DD. Got {date!r}.")
    if venue and not _VENUE_RE.match(venue):
        raise ValueError(f"venue must be a short code like 'WFM'. Got {venue!r}.")
    if not _JURISDICTION_RE.match(jurisdiction):
        raise ValueError(f"jurisdiction must be a state code. Got {jurisdiction!r}.")


# TAB's edge rejects plain API clients from some networks, so present as a
# browser. Set TAB_FIXTURES=1 to read the saved responses in evals/fixtures
# instead of the network — that's what makes the workshop reproducible, and
# what to fall back on if the live endpoint is unreachable.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Origin": "https://www.tab.com.au",
    "Referer": "https://www.tab.com.au/",
}

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures"

# The originally captured files, all of them Warwick Farm R2 on 2026-09-23.
# They are only used when they actually match what was asked for; to add more
# races, save them under the naming convention in _fixture_candidates().
GENERIC_FIXTURES = {
    "meetings": "tab_meetings_by_jurisdiction_info.json",
    "race": "tab_racing_info.json",
    "form": "tab_form_by_runner.json",
    "race_form": "tab_form_by_race.json",
}

_SELF_LINK_RE = re.compile(r"/dates/(\d{4}-\d{2}-\d{2})/meetings/\w+/(\w+)/races/(\d+)")


def use_fixtures() -> bool:
    return os.getenv("TAB_FIXTURES", "").lower() in {"1", "true", "yes"}


def _fixture_identity(payload: dict, kind: str) -> tuple[str, str, int] | None:
    """What race a saved payload is actually for: (date, venue, race_number)."""
    if kind == "race":
        meeting = payload.get("meeting") or {}
        if meeting.get("meetingDate") and meeting.get("venueMnemonic"):
            return (
                meeting["meetingDate"],
                meeting["venueMnemonic"],
                payload.get("raceNumber"),
            )
        return None

    # The form payloads carry their own URL, which names the race.
    entries = payload.get("form") if kind == "race_form" else [payload]
    link = ((entries or [{}])[0].get("_links") or {}).get("self", "")
    match = _SELF_LINK_RE.search(link)
    return (match[1], match[2], int(match[3])) if match else None


def _load_fixture(kind: str, date: str, venue: str, race_number: int | None) -> dict:
    """Return the saved response for this exact race, or say what's missing.

    Never substitutes a different race: a fixture that quietly answers for the
    wrong runners is worse than no fixture at all.
    """
    venue = (venue or "").upper()
    specific = {
        "race": f"tab_race_{venue}_{date}_R{race_number}.json",
        "race_form": f"tab_form_{venue}_{date}_R{race_number}.json",
        "form": f"tab_form_{venue}_{date}_R{race_number}.json",
    }.get(kind)

    if specific and (FIXTURE_DIR / specific).is_file():
        return json.loads((FIXTURE_DIR / specific).read_text())

    payload = json.loads((FIXTURE_DIR / GENERIC_FIXTURES[kind]).read_text())

    # The meetings file covers every meeting that day, so only the date matters.
    if kind == "meetings":
        return payload

    identity = _fixture_identity(payload, kind)
    if identity == (date, venue, race_number):
        return payload

    have = (
        f"{identity[1]} R{identity[2]} on {identity[0]}" if identity else "an unknown race"
    )
    raise ValueError(
        f"TAB_FIXTURES is on and there is no saved response for {venue} R{race_number} "
        f"on {date} — the saved file holds {have}. Either save that race to "
        f"evals/fixtures/{specific}, or unset TAB_FIXTURES to use the live API."
    )


def _http_get(url: str, params: dict, timeout: float) -> dict:
    """One GET against the live API.

    TAB sits behind a WAF that rejects ordinary API clients on TLS fingerprint
    alone — curl and httpx time out where a browser gets a 200. curl_cffi
    impersonates Chrome's handshake, which is what makes live calls work at
    all. httpx stays as a fallback for environments without it.
    """
    if curl_requests is not None:
        try:
            response = curl_requests.get(
                url,
                params=params,
                # Override with TAB_IMPERSONATE (e.g. "chrome124", "safari17_0")
                # if the WAF stops accepting the default profile.
                impersonate=os.getenv("TAB_IMPERSONATE", "chrome"),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors vary by backend
            raise ValueError(f"Could not reach the TAB API: {exc}") from exc
        if response.status_code >= 400:
            raise ValueError(
                f"TAB returned {response.status_code}. Check the date, venue "
                "code and race number."
            )
        return response.json()

    response = httpx.get(
        url, params=params, headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True
    )
    response.raise_for_status()
    return response.json()


def _get(
    url: str,
    params: dict,
    timeout: float,
    fixture: str,
    date: str = "",
    venue: str = "",
    race_number: int | None = None,
) -> dict:
    """One GET, or the saved response when TAB_FIXTURES is set."""
    if use_fixtures():
        return _load_fixture(fixture, date, venue, race_number)
    return _http_get(url, params, timeout)


def fetch_meetings(date: str, jurisdiction: str = "NSW", timeout: float = 20.0) -> dict:
    """GET the meetings for a date. Returns the raw payload — normalise it next."""
    _check(date, "", jurisdiction)
    url = f"{API_ROOT}/dates/{date}/meetings/"
    params = {"jurisdiction": jurisdiction.upper()}
    return _get(url, params, timeout, "meetings", date=date)


# The meetings payload is big and slow-changing; one fetch per date serves
# every venue check for that day.
_MEETINGS_CACHE: dict[tuple, dict] = {}


def venues_on(
    date: str, race_type: str = "R", jurisdiction: str = "NSW"
) -> dict[str, dict]:
    """Every venue racing on a date, keyed by venue code, across all locations.

    Used to check a venue code before building a race URL — the agent should
    never be able to guess one that silently resolves to another track.

    Codes are only unique within a race type: Warrnambool's gallops and dogs
    are both WBO. Filter first, or the last meeting with a code overwrites
    the rest and a valid thoroughbred code is rejected.
    """
    key = (date, jurisdiction.upper())
    if key not in _MEETINGS_CACHE:
        _MEETINGS_CACHE[key] = fetch_meetings(date, jurisdiction)
    return {
        m["venueMnemonic"]: m
        for m in (_MEETINGS_CACHE[key].get("meetings") or [])
        if m.get("venueMnemonic") and m.get("raceType") == race_type
    }


# Nicknames that share no words with the official name. Anything that is a
# prefix or substring of the real name is handled by the tiers below instead.
VENUE_ALIASES = {
    "the valley": "moonee valley",
    "headquarters": "flemington",
    "hq": "flemington",
}

_COUNTRY_SUFFIX = re.compile(r"\s*\([A-Za-z]{2,4}\)\s*$")


def _normalise_venue(name: str) -> str:
    """Lowercase, drop the country suffix, squeeze punctuation and spacing."""
    name = _COUNTRY_SUFFIX.sub("", name or "").lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return " ".join(name.split())


def resolve_venue(date: str, query: str, race_type: str = "R") -> dict:
    """Find the one meeting a user means, or ask them which.

    Searches every meeting that day, including ones whose card has already
    been run — a finished meeting that is filtered out of the list is exactly
    how "Horseshoe Indianapolis" got answered with "Horseshoe Indianapolis
    Extra". Matching runs in tiers (exact, prefix, substring) and the first
    tier with any hit wins, so "Woodbine" resolves to Woodbine and not to
    Woodbine Extra. More than one hit is a question for the user, never a
    guess — including when the candidates are in different countries.
    """
    key = (date, "NSW")
    if key not in _MEETINGS_CACHE:
        _MEETINGS_CACHE[key] = fetch_meetings(date)
    meetings = [
        m
        for m in _MEETINGS_CACHE[key].get("meetings") or []
        if m.get("raceType") == race_type
    ]

    wanted = _normalise_venue(query)
    wanted = VENUE_ALIASES.get(wanted, wanted)
    if not wanted:
        raise ValueError("Which venue? Give me a track name.")

    named = [(m, _normalise_venue(m.get("meetingName", ""))) for m in meetings]
    for tier in (
        lambda n: n == wanted,
        lambda n: n.startswith(wanted),
        lambda n: wanted in n,
    ):
        hits = [m for m, n in named if tier(n)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            options = ", ".join(
                f"{m['meetingName']} ({m.get('location')})" for m in hits
            )
            raise ValueError(
                f"More than one meeting matches {query!r} on {date}: {options}. "
                "Which one do you mean?"
            )

    racing = ", ".join(sorted(m["meetingName"] for m in meetings))
    raise ValueError(
        f"No meeting matching {query!r} on {date}. Racing that day: {racing}."
    )


def check_venue(date: str, venue: str, race_type: str = "R") -> dict:
    """Confirm a venue is racing on this date, or list the ones that are."""
    venues = venues_on(date, race_type)
    meeting = venues.get(venue.upper())
    if meeting:
        return meeting
    valid = sorted(
        f"{m['venueMnemonic']} ({m.get('meetingName')}, {m.get('location')})"
        for m in venues.values()
    )
    raise ValueError(
        f"No {race_type} meeting at {venue.upper()} on {date}. "
        f"Racing that day: {', '.join(valid) if valid else 'nothing found'}."
    )


def fetch_race(
    date: str,
    venue: str,
    race_number: int,
    race_type: str = "R",
    jurisdiction: str = "NSW",
    timeout: float = 20.0,
) -> dict:
    """GET one race. Returns the raw payload — normalise it next."""
    _check(date, venue, jurisdiction)
    if not isinstance(race_number, int) or not 1 <= race_number <= 20:
        raise ValueError(f"race_number must be 1-20. Got {race_number!r}.")
    url = f"{API_ROOT}/dates/{date}/meetings/{race_type}/{venue.upper()}/races/{race_number}"
    params = {"jurisdiction": jurisdiction.upper(), "returnOffers": "true", "returnPromo": "true"}
    return _get(url, params, timeout, "race", date=date, venue=venue, race_number=race_number)


# Form is history: it doesn't change while we're looking at it, so cache it
# for the session. Race payloads are deliberately NOT cached — odds move.
_FORM_CACHE: dict[tuple, dict] = {}


def clear_form_cache() -> None:
    """Drop the cached form payloads."""
    _FORM_CACHE.clear()


def fetch_form(
    date: str,
    venue: str,
    race_number: int,
    runner_number: int,
    race_type: str = "R",
    jurisdiction: str = "NSW",
    timeout: float = 20.0,
) -> dict:
    """GET one runner's form. Returns the raw payload — normalise it next.

    Cached, so ranking the field and then drilling into one runner costs a
    single request per horse rather than two.
    """
    key = (date, venue.upper(), race_number, runner_number)
    if key in _FORM_CACHE:
        return _FORM_CACHE[key]
    _check(date, venue, jurisdiction)
    if not isinstance(race_number, int) or not 1 <= race_number <= 20:
        raise ValueError(f"race_number must be 1-20. Got {race_number!r}.")
    if not isinstance(runner_number, int) or not 1 <= runner_number <= 30:
        raise ValueError(f"runner_number must be 1-30. Got {runner_number!r}.")
    url = (
        f"{API_ROOT}/dates/{date}/meetings/{race_type}/{venue.upper()}"
        f"/races/{race_number}/form/{runner_number}"
    )
    print()
    print(url)
    print()
    params = {"jurisdiction": jurisdiction.upper()}
    payload = _get(url, params, timeout, "form", date=date, venue=venue, race_number=race_number)
    _FORM_CACHE[key] = payload
    return payload


def fetch_race_form(
    date: str,
    venue: str,
    race_number: int,
    race_type: str = "R",
    jurisdiction: str = "NSW",
    timeout: float = 30.0,
) -> dict[int, dict]:
    """GET the whole field's form in one request, keyed by runner number.

    Drop the runner number from the form URL and the endpoint returns every
    runner, in the same per-runner shape. Each one is dropped into the cache
    on the way past, so a later drill-down on any of them is free.
    """
    _check(date, venue, jurisdiction)
    if not isinstance(race_number, int) or not 1 <= race_number <= 20:
        raise ValueError(f"race_number must be 1-20. Got {race_number!r}.")
    url = (
        f"{API_ROOT}/dates/{date}/meetings/{race_type}/{venue.upper()}"
        f"/races/{race_number}/form/"
    )
    print("")
    print(url)
    params = {"jurisdiction": jurisdiction.upper()}
    payload = _get(url, params, timeout, "race_form", date=date, venue=venue, race_number=race_number)

    field: dict[int, dict] = {}
    for entry in payload.get("form") or []:
        number = entry.get("runnerNumber")
        if number is None:
            continue
        field[number] = entry
        _FORM_CACHE[(date, venue.upper(), race_number, number)] = entry
    return field

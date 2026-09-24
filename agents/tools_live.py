"""Tools backed by the live TAB info service.

The same four tools as agents/tools.py, against real racing instead of the
fixtures:

    list_meetings      what's on today, and what the ground is doing
    get_race           one race: field, going, prices, attributed editorial
    get_runner_form    one runner's history, when the field summary isn't enough
    recommend_runners  our ranking, and where it disagrees with the market

agents/tools.py stays on the frozen fixtures so the evals keep comparing like
with like. Swap TOOLS for TOOLS_LIVE in agents/agent.py to run on live data.
"""

from __future__ import annotations

from datetime import date as _date

import httpx
from langchain_core.tools import tool

from agents.mock_data import going_distance
from agents.tab_api import (
    check_venue,
    fetch_form,
    fetch_meetings,
    fetch_race,
    fetch_race_form,
    normalize_form,
    normalize_meetings,
    normalize_race,
    resolve_venue,
)

# Ground where a wet-track record is worth something.
WET_GOING = {"Good To Soft", "Soft", "Heavy"}

# A record of four starts or more is treated as a full-strength sample.
# Below that the strike rate is discounted rather than trusted outright —
# 1 win from 1 start is not a 100% horse.
FULL_CONFIDENCE_STARTS = 4


def _today() -> str:
    return _date.today().isoformat()


def _call(fn, *args, **kwargs):
    """Run a fetch, turning transport failures into something the model can act on."""
    try:
        return fn(*args, **kwargs)
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"TAB returned {exc.response.status_code} for that request. "
            "Check the date, venue code and race number."
        ) from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"Could not reach the TAB API: {exc}") from exc


@tool
def list_meetings(
    date: str | None = None, state: str = "NSW", venue: str | None = None
) -> list[dict]:
    """List today's thoroughbred meetings, with the going and the races still to run.

    Args:
        date: Race day as YYYY-MM-DD. Defaults to today.
        state: Australian state, e.g. NSW, VIC, QLD. Use "ALL" for every location.
        venue: Optional venue name or code to search across all locations.
    """
    day = date or _today()
    payload = _call(fetch_meetings, day)
    if venue:
        meeting = resolve_venue(payload, venue)
        payload = {"meetings": [meeting]}
        state = meeting.get("location") or state
        location = None
    else:
        location = None if state.upper() == "ALL" else state.upper()
    meetings = normalize_meetings(payload, state=location)
    if not meetings:
        raise ValueError(
            f"No thoroughbred meetings still to run in {state} on {day}. "
            "Try another state, or ALL."
        )
    return meetings


@tool
def get_race(date: str, venue_code: str, race_number: int) -> dict:
    """Get one race: the field, the going, the prices and the published previews.

    Every runner carries its wet-track record, form rating, barrier and early
    speed. Scratched runners are already removed.

    Args:
        date: Race day as YYYY-MM-DD.
        venue_code: Short venue code from list_meetings, e.g. WFM.
        race_number: Race number on the card.
    """
    _call(check_venue, date, venue_code)
    payload = _call(fetch_race, date, venue_code, race_number)
    return normalize_race(payload)


@tool
def get_runner_form(
    date: str, venue_code: str, race_number: int, runner_number: int
) -> dict:
    """Get one runner's record, broken down by going, track, distance and class.

    Use this on a shortlisted runner when you need real numbers behind a
    hunch — how many times it has actually run on Soft, and how it went. It is
    one request per horse, so don't call it on the whole field.

    Args:
        date: Race day as YYYY-MM-DD.
        venue_code: Short venue code, e.g. WFM.
        race_number: Race number on the card.
        runner_number: Saddlecloth number of the runner.
    """
    payload = _call(fetch_form, date, venue_code, race_number, runner_number)
    return normalize_form(payload)


def _going_points(by_going: dict, going: str) -> tuple[float, str]:
    """Score a runner's record on today's ground, and on ground either side of it.

    Exact matches are rare: a Soft5 track is "Good To Soft" on our scale, but
    the form endpoint files those runs under `soft` (or legacy `slow`). So
    neighbouring going counts too, at half weight — a horse's Soft record is
    real evidence for a Good To Soft track, just not as direct.

    A strike rate alone would rate 1-from-1 as a certainty, so the result is
    scaled by how much evidence sits behind it.
    """
    starts = wins = placings = 0.0
    used: list[str] = []

    for bucket, record in (by_going or {}).items():
        try:
            steps = going_distance(bucket, going)
        except ValueError:  # a going we don't recognise
            continue
        if steps > 1 or not record["starts"]:
            continue
        weight = 1.0 if steps == 0 else 0.5
        starts += weight * record["starts"]
        wins += weight * record["wins"]
        placings += weight * record["placings"]
        used.append(f"{record['wins']}/{record['starts']} on {bucket}")

    if not starts:
        return 0.0, f"no record on {going} or near it"

    confidence = min(starts, FULL_CONFIDENCE_STARTS) / FULL_CONFIDENCE_STARTS
    points = (5 * wins / starts + 1.5 * placings / starts) * confidence
    return points, f"{', '.join(used)} (today is {going})"


@tool
def recommend_runners(
    date: str, venue_code: str, race_number: int, use_form: bool | None = None
) -> dict:
    """Rank a field for today's going, and show where we disagree with the market.

    Pulls every runner's actual record on today's going — real starts and
    wins, not a wet/dry flag — then compares the ranking against the price the
    market is offering. A positive value_gap means the horse rates better than
    its odds suggest.

    Args:
        date: Race day as YYYY-MM-DD.
        venue_code: Short venue code from list_meetings, e.g. WFM.
        race_number: Race number on the card.
        use_form: Pull the field's full records. On by default — it is one
            request for the whole field — set False to rank on the race card
            alone.
    """
    _call(check_venue, date, venue_code)
    race = normalize_race(_call(fetch_race, date, venue_code, race_number))
    going = race["meeting"]["going"]
    wet = going in WET_GOING

    # One request covers the whole field, and seeds the cache, so a later
    # get_runner_form on any of these runners costs nothing.
    forms: dict[int, dict] = {}
    if going and (use_form if use_form is not None else True):
        try:
            field = fetch_race_form(date, venue_code, race_number)
            forms = {n: normalize_form(p) for n, p in field.items()}
        except Exception:  # noqa: BLE001 - rank on the race card rather than fail
            forms = {}

    scored = []
    for r in race["runners"]:
        reasons = []
        score = 0.0
        form = forms.get(r["number"])

        if form:
            points, why = _going_points(form["record"]["by_going"], going)
            score += points
            reasons.append(why)
        elif wet and r["wet_winner"]:
            # No detailed record for this runner — fall back to the race card.
            score += 3
            reasons.append(f"listed as a wet-track winner; today is {going} (no detailed record)")
        elif wet and r["wet_placed"]:
            score += 1.5
            reasons.append(f"listed as wet-track placed; today is {going} (no detailed record)")
        elif wet:
            reasons.append(f"no wet-track record and today is {going}")
        elif r["track_winner"]:
            # Dry or unknown going: the card still tells us it handles the track.
            score += 1
            reasons.append("listed as a winner at this track")
        else:
            reasons.append(
                f"going is {going or 'not published for this meeting'}; "
                "ranked on form rating and course/distance record"
            )

        # Form rating is 0-100, so this is worth up to 4 points.
        if r["form_rating"]:
            score += r["form_rating"] / 25
            reasons.append(f"form rating {r['form_rating']}")

        if r["course_winner"]:
            score += 1
            reasons.append("previous winner at this course")
        if r["distance_winner"]:
            score += 1
            reasons.append("previous winner at this trip")

        scored.append({**r, "score": round(score, 1), "reasons": reasons})

    # Deliberately NOT a probability. Scores here span about 1.5x across a
    # field the market spreads 39x, so normalising them into percentages
    # produced a "value gap" on every longshot — an artefact of the scale,
    # not an edge. We publish the order and the evidence, put the market's
    # own view beside it, and let the difference be a difference in RANK.
    scored.sort(key=lambda s: s["score"], reverse=True)
    by_market = sorted(
        [s for s in scored if s.get("implied_win_pct") is not None],
        key=lambda s: s["implied_win_pct"],
        reverse=True,
    )
    market_rank = {s["number"]: i + 1 for i, s in enumerate(by_market)}

    ranking = []
    for position, s in enumerate(scored, start=1):
        m_rank = market_rank.get(s["number"])
        gap = m_rank - position if m_rank else None
        ranking.append(
            {
                "rank": position,
                "number": s["number"],
                "name": s["name"],
                "score": s["score"],
                "market_rank": m_rank,
                "rates_higher_than_market_by": gap,
                "win_odds": s.get("win_odds"),
                "market_move": s.get("market_move"),
                "reasons": s["reasons"],
            }
        )

    return {
        "race": race["race"]["name"],
        "venue": race["meeting"]["venue"],
        "going": going,
        "ranked_on": (
            "record on today's going" if forms
            else "form rating only — no track condition published for this meeting"
            if not going
            else "race-card form flags — detailed records unavailable"
        ),
        "race_shape": race["race_shape"],
        "odds_as_at": race["odds_as_at"],
        "ranking": ranking,
        "caveat": (
            (
                "" if going else
                "NO TRACK CONDITION IS PUBLISHED FOR THIS MEETING, so nothing here "
                "reflects the going — say so before giving a pick. "
            )
            + "This ranking reflects each runner's record on today's going, its "
            "form rating and its course/distance record. It is an order of "
            "preference, not a probability, and it does not price fitness, "
            "class, the speed map or market money. Where it disagrees with the "
            "price, say so as a difference of opinion, not as an edge."
        ),
        # Other people's opinions. Name them if you use them.
        "third_party": race["editorial"],
    }


TOOLS_LIVE = [list_meetings, get_race, get_runner_form, recommend_runners]

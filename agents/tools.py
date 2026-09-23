"""Tools available to the horse recommender agent.

Four tools, one per step of the reasoning the agent has to do:

    list_races        what is on
    get_race          the course and the weather, so we know today's going
    get_horse         the horse, its yard, and how it has run on that going
    recommend_runners rank the field for the conditions

TODO: add a documentation search tool once retrieval is back.
"""

from __future__ import annotations

from langchain_core.tools import tool

from agents.mock_data import (
    HORSES,
    RACE_COURSES,
    TRAINERS,
    UPCOMING_RACES,
    going_distance,
    horse_form,
    weather_for,
)


@tool
def list_races() -> list[dict]:
    """List the upcoming races, with the course and distance for each."""
    return [
        {
            "race_id": race["race_id"],
            "name": race["name"],
            "date": race["date"],
            "course": RACE_COURSES[race["course_id"]]["name"],
            "distance_furlongs": race["distance_furlongs"],
            "runners": len(race["runners"]),
        }
        for race in UPCOMING_RACES.values()
    ]


@tool
def get_race(race_id: str) -> dict:
    """Get a race, the course it is run at, and the weather and going expected on the day.

    Args:
        race_id: The race ID, e.g. RC-THW-01. Use list_races to find one.
    """
    race = _require(UPCOMING_RACES, race_id, "race", "RC-XXX-##")
    course = RACE_COURSES[race["course_id"]]
    weather = weather_for(course["course_id"], race["date"])
    return {
        "race_id": race["race_id"],
        "name": race["name"],
        "date": race["date"],
        "distance_furlongs": race["distance_furlongs"],
        "course": course["name"],
        "course_notes": course["notes"],
        "going": weather["going"] if weather else "unknown",
        "weather": weather["condition"] if weather else "no reading",
        "wind_kph": weather["wind_kph"] if weather else None,
        "runners": [
            {"horse_id": hid, "name": HORSES[hid]["name"]} for hid in race["runners"]
        ],
    }


@tool
def get_horse(horse_id: str) -> dict:
    """Get a horse, its trainer, and its past runs.

    Each past run records the going and weather it was run on, so use this to
    judge how the horse handles today's conditions.

    Args:
        horse_id: The horse ID, e.g. HRS-1001.
    """
    horse = _require(HORSES, horse_id, "horse", "HRS-####")
    return {
        **horse,
        "trainer": TRAINERS[horse["trainer_id"]]["name"],
        "form": [
            {
                "date": run["date"],
                "course": RACE_COURSES[run["course_id"]]["name"],
                "going": run["going"],
                "weather": run["condition"],
                "distance_furlongs": run["distance_furlongs"],
                "finished": f"{run['finish_position']} of {run['field_size']}",
            }
            for run in horse_form(horse_id)
        ],
    }


@tool
def recommend_runners(race_id: str) -> dict:
    """Rank the runners in a race by how well they suit the going expected on the day.

    Args:
        race_id: The race ID, e.g. RC-THW-01. Use list_races to find one.
    """
    race = _require(UPCOMING_RACES, race_id, "race", "RC-XXX-##")
    weather = weather_for(race["course_id"], race["date"])
    going = weather["going"] if weather else "Good"

    picks = []
    for horse_id in race["runners"]:
        horse = HORSES[horse_id]
        # How far today's ground is from the ground this horse wants.
        steps_off = going_distance(horse["preferred_going"], going)
        # Its record on ground within one step of today's.
        similar = [r for r in horse_form(horse_id) if going_distance(r["going"], going) <= 1]
        wins = sum(1 for r in similar if r["finish_position"] == 1)
        picks.append(
            {
                "name": horse["name"],
                "score": (5 - steps_off) + 2 * wins,
                "reason": (
                    f"wants {horse['preferred_going']}, today is {going}; "
                    f"{wins} win(s) from {len(similar)} run(s) on similar ground"
                ),
            }
        )

    picks.sort(key=lambda p: p["score"], reverse=True)
    return {
        "race": race["name"],
        "going": going,
        "ranking": picks,
    }


def _require(table: dict, key: str, label: str, fmt: str) -> dict:
    """Look up a key, or raise an error the model can act on."""
    if key not in table:
        raise ValueError(
            f"No {label} found with ID {key!r}. {label.title()} IDs look like {fmt}."
        )
    return table[key]


TOOLS = [list_races, get_race, get_horse, recommend_runners]

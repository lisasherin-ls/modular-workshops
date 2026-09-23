"""In-memory fake racing data used by the horse recommender tools.

Everything here is invented for the workshop: no real horse, person, venue or
race. Five tables, wired together so a recommendation can be derived:

    UpcomingRace -> RaceCourse -> WeatherObservation (gives today's going)
                 -> Horse -> Trainer
                          -> PastRun (one per previous start, each carrying
                             the going it was run on)
"""

from __future__ import annotations

from typing import TypedDict


# The going scale, driest to wettest. Positions on this list let us ask how
# far today's ground is from the ground a horse wants.
GOING_SCALE: list[str] = [
    "Firm",
    "Good To Firm",
    "Good",
    "Good To Soft",
    "Soft",
    "Heavy",
]


class Trainer(TypedDict):
    trainer_id: str
    name: str
    yard: str
    strike_rate: float
    notes: str


class Horse(TypedDict):
    horse_id: str
    name: str
    age: int
    trainer_id: str
    official_rating: int
    preferred_going: str
    notes: str


class RaceCourse(TypedDict):
    course_id: str
    name: str
    city: str
    surface: str
    notes: str


class WeatherObservation(TypedDict):
    course_id: str
    date: str
    condition: str
    temp_c: float
    wind_kph: float
    rain_mm_24h: float
    going: str


class PastRun(TypedDict):
    horse_id: str
    course_id: str
    date: str
    distance_furlongs: int
    going: str
    condition: str
    finish_position: int
    field_size: int


class UpcomingRace(TypedDict):
    race_id: str
    name: str
    course_id: str
    date: str
    distance_furlongs: int
    runners: list[str]


TRAINERS: dict[str, Trainer] = {
    "TRN-001": {
        "trainer_id": "TRN-001",
        "name": "Fiona Marchetti",
        "yard": "Coldharbour Stables",
        "strike_rate": 0.19,
        "notes": "Strong record with mud-loving stayers.",
    },
    "TRN-002": {
        "trainer_id": "TRN-002",
        "name": "Dermot Ashworth",
        "yard": "Ivy Gate Racing",
        "strike_rate": 0.13,
        "notes": "Speed yard. Sprinters that flatter on fast ground and struggle once it rains.",
    },
    "TRN-003": {
        "trainer_id": "TRN-003",
        "name": "Aoife Brennan",
        "yard": "Carrigmore Lodge",
        "strike_rate": 0.23,
        "notes": "Small string, high strike rate; targets soft-ground handicaps.",
    },
    "TRN-004": {
        "trainer_id": "TRN-004",
        "name": "Peter Nakamura",
        "yard": "Hollow Ridge",
        "strike_rate": 0.09,
        "notes": "Volume yard, inconsistent.",
    },
}


HORSES: dict[str, Horse] = {
    "HRS-1001": {
        "horse_id": "HRS-1001",
        "name": "Copper Lantern",
        "age": 5,
        "trainer_id": "TRN-001",
        "official_rating": 94,
        "preferred_going": "Soft",
        "notes": "Genuine mudlark. Stops in the last furlong on quick ground.",
    },
    "HRS-1002": {
        "horse_id": "HRS-1002",
        "name": "Vellum Sky",
        "age": 4,
        "trainer_id": "TRN-002",
        "official_rating": 101,
        "preferred_going": "Good To Firm",
        "notes": "Top-end speed on a sound surface; has never completed on Soft or worse.",
    },
    "HRS-1003": {
        "horse_id": "HRS-1003",
        "name": "Harrow Bell",
        "age": 6,
        "trainer_id": "TRN-003",
        "official_rating": 97,
        "preferred_going": "Good To Soft",
        "notes": "Handles most surfaces without ever quite winning.",
    },
    "HRS-1004": {
        "horse_id": "HRS-1004",
        "name": "Salt Marsh Ruby",
        "age": 4,
        "trainer_id": "TRN-001",
        "official_rating": 88,
        "preferred_going": "Heavy",
        "notes": "Stamina type, needs a real test. Unraced on faster than Good.",
    },
    "HRS-1005": {
        "horse_id": "HRS-1005",
        "name": "Quarry Light",
        "age": 7,
        "trainer_id": "TRN-004",
        "official_rating": 82,
        "preferred_going": "Good",
        "notes": "Veteran. Consistent at Ellerby, well beaten everywhere else.",
    },
    "HRS-1006": {
        "horse_id": "HRS-1006",
        "name": "Nine Pennies",
        "age": 3,
        "trainer_id": "TRN-002",
        "official_rating": 91,
        "preferred_going": "Firm",
        "notes": "Lightly raced sprinter, only three starts.",
    },
    "HRS-1007": {
        "horse_id": "HRS-1007",
        "name": "Gale Warning",
        "age": 5,
        "trainer_id": "TRN-003",
        "official_rating": 99,
        "preferred_going": "Soft",
        "notes": "Unbothered by wind and rain.",
    },
}


RACE_COURSES: dict[str, RaceCourse] = {
    "CRS-ELL": {
        "course_id": "CRS-ELL",
        "name": "Ellerby Park",
        "city": "Malton",
        "surface": "turf",
        "notes": "Galloping oval. Dries out quickly after rain.",
    },
    "CRS-THW": {
        "course_id": "CRS-THW",
        "name": "Thornwick Downs",
        "city": "Lambourn",
        "surface": "turf",
        "notes": "Drains poorly, rides heavy in winter. Stiff uphill finish.",
    },
    "CRS-CAS": {
        "course_id": "CRS-CAS",
        "name": "Castlereagh",
        "city": "Kildare",
        "surface": "turf",
        "notes": "Exposed; the wind off the plain is a genuine factor.",
    },
    "CRS-BRK": {
        "course_id": "CRS-BRK",
        "name": "Brackenhall",
        "city": "Newmarket",
        "surface": "turf",
        "notes": "Straight sprint track, the fastest surface in the fixture list.",
    },
}


WEATHER: list[WeatherObservation] = [
    {"course_id": "CRS-ELL", "date": "2026-09-26", "condition": "showers", "temp_c": 12.5, "wind_kph": 22.0, "rain_mm_24h": 11.0, "going": "Good To Soft"},
    {"course_id": "CRS-THW", "date": "2026-09-26", "condition": "drizzle", "temp_c": 11.5, "wind_kph": 24.0, "rain_mm_24h": 14.0, "going": "Heavy"},
    {"course_id": "CRS-CAS", "date": "2026-09-26", "condition": "windy", "temp_c": 12.5, "wind_kph": 38.0, "rain_mm_24h": 9.0, "going": "Soft"},
    {"course_id": "CRS-BRK", "date": "2026-09-26", "condition": "clear", "temp_c": 19.0, "wind_kph": 9.0, "rain_mm_24h": 0.8, "going": "Good To Firm"},
]


# Past starts. The going is stored on the run itself, not looked up from the
# weather table: what matters is the ground the race was actually run on.
PAST_RUNS: list[PastRun] = [
    # Copper Lantern -- clear soft-ground profile.
    {"horse_id": "HRS-1001", "course_id": "CRS-THW", "date": "2026-08-12", "distance_furlongs": 12, "going": "Soft", "condition": "light rain", "finish_position": 1, "field_size": 11},
    {"horse_id": "HRS-1001", "course_id": "CRS-BRK", "date": "2026-07-19", "distance_furlongs": 10, "going": "Firm", "condition": "sunny", "finish_position": 8, "field_size": 9},
    {"horse_id": "HRS-1001", "course_id": "CRS-ELL", "date": "2026-06-28", "distance_furlongs": 12, "going": "Good To Soft", "condition": "overcast", "finish_position": 2, "field_size": 10},
    {"horse_id": "HRS-1001", "course_id": "CRS-THW", "date": "2026-05-03", "distance_furlongs": 14, "going": "Heavy", "condition": "heavy rain", "finish_position": 1, "field_size": 8},
    # Vellum Sky -- fast-ground specialist, falls apart when it rains.
    {"horse_id": "HRS-1002", "course_id": "CRS-BRK", "date": "2026-08-30", "distance_furlongs": 6, "going": "Good To Firm", "condition": "sunny", "finish_position": 1, "field_size": 12},
    {"horse_id": "HRS-1002", "course_id": "CRS-ELL", "date": "2026-08-05", "distance_furlongs": 7, "going": "Good", "condition": "cloudy", "finish_position": 2, "field_size": 9},
    {"horse_id": "HRS-1002", "course_id": "CRS-CAS", "date": "2026-07-11", "distance_furlongs": 6, "going": "Soft", "condition": "squalls", "finish_position": 9, "field_size": 10},
    {"horse_id": "HRS-1002", "course_id": "CRS-BRK", "date": "2026-06-02", "distance_furlongs": 6, "going": "Firm", "condition": "clear", "finish_position": 1, "field_size": 8},
    # Harrow Bell -- versatile, rarely wins.
    {"horse_id": "HRS-1003", "course_id": "CRS-CAS", "date": "2026-09-01", "distance_furlongs": 10, "going": "Soft", "condition": "windy", "finish_position": 3, "field_size": 12},
    {"horse_id": "HRS-1003", "course_id": "CRS-ELL", "date": "2026-07-14", "distance_furlongs": 10, "going": "Good To Firm", "condition": "sunny", "finish_position": 3, "field_size": 10},
    {"horse_id": "HRS-1003", "course_id": "CRS-THW", "date": "2026-06-21", "distance_furlongs": 12, "going": "Good To Soft", "condition": "drizzle", "finish_position": 1, "field_size": 9},
    # Salt Marsh Ruby -- has never run on faster than Good.
    {"horse_id": "HRS-1004", "course_id": "CRS-THW", "date": "2026-08-24", "distance_furlongs": 14, "going": "Heavy", "condition": "heavy rain", "finish_position": 1, "field_size": 7},
    {"horse_id": "HRS-1004", "course_id": "CRS-CAS", "date": "2026-07-29", "distance_furlongs": 16, "going": "Soft", "condition": "light rain", "finish_position": 2, "field_size": 9},
    {"horse_id": "HRS-1004", "course_id": "CRS-ELL", "date": "2026-06-10", "distance_furlongs": 12, "going": "Good", "condition": "overcast", "finish_position": 6, "field_size": 10},
    # Quarry Light -- a course specialist rather than a going specialist.
    {"horse_id": "HRS-1005", "course_id": "CRS-ELL", "date": "2026-09-05", "distance_furlongs": 8, "going": "Good", "condition": "cloudy", "finish_position": 1, "field_size": 10},
    {"horse_id": "HRS-1005", "course_id": "CRS-ELL", "date": "2026-08-17", "distance_furlongs": 7, "going": "Good To Firm", "condition": "sunny", "finish_position": 2, "field_size": 8},
    {"horse_id": "HRS-1005", "course_id": "CRS-BRK", "date": "2026-07-26", "distance_furlongs": 7, "going": "Firm", "condition": "sunny", "finish_position": 11, "field_size": 12},
    # Nine Pennies -- only three starts, deliberately thin evidence.
    {"horse_id": "HRS-1006", "course_id": "CRS-BRK", "date": "2026-09-03", "distance_furlongs": 5, "going": "Firm", "condition": "sunny", "finish_position": 1, "field_size": 9},
    {"horse_id": "HRS-1006", "course_id": "CRS-BRK", "date": "2026-08-10", "distance_furlongs": 6, "going": "Good To Firm", "condition": "clear", "finish_position": 3, "field_size": 10},
    {"horse_id": "HRS-1006", "course_id": "CRS-ELL", "date": "2026-07-15", "distance_furlongs": 6, "going": "Good", "condition": "cloudy", "finish_position": 4, "field_size": 12},
    # Gale Warning -- wins in the wet and the wind.
    {"horse_id": "HRS-1007", "course_id": "CRS-CAS", "date": "2026-09-08", "distance_furlongs": 12, "going": "Soft", "condition": "squalls", "finish_position": 1, "field_size": 11},
    {"horse_id": "HRS-1007", "course_id": "CRS-CAS", "date": "2026-08-15", "distance_furlongs": 10, "going": "Good To Soft", "condition": "windy", "finish_position": 1, "field_size": 9},
    {"horse_id": "HRS-1007", "course_id": "CRS-ELL", "date": "2026-07-22", "distance_furlongs": 10, "going": "Good", "condition": "sunny", "finish_position": 5, "field_size": 10},
    {"horse_id": "HRS-1007", "course_id": "CRS-THW", "date": "2026-06-19", "distance_furlongs": 12, "going": "Soft", "condition": "heavy rain", "finish_position": 2, "field_size": 12},
]


UPCOMING_RACES: dict[str, UpcomingRace] = {
    "RC-THW-01": {
        "race_id": "RC-THW-01",
        "name": "Thornwick Autumn Stayers' Handicap",
        "course_id": "CRS-THW",
        "date": "2026-09-26",
        "distance_furlongs": 14,
        "runners": ["HRS-1001", "HRS-1003", "HRS-1004", "HRS-1007"],
    },
    "RC-BRK-01": {
        "race_id": "RC-BRK-01",
        "name": "Brackenhall Sprint Trophy",
        "course_id": "CRS-BRK",
        "date": "2026-09-26",
        "distance_furlongs": 6,
        "runners": ["HRS-1002", "HRS-1006", "HRS-1005"],
    },
    "RC-CAS-01": {
        "race_id": "RC-CAS-01",
        "name": "Castlereagh Windward Stakes",
        "course_id": "CRS-CAS",
        "date": "2026-09-26",
        "distance_furlongs": 12,
        "runners": ["HRS-1001", "HRS-1007", "HRS-1004"],
    },
}


def horse_form(horse_id: str) -> list[PastRun]:
    """Return a horse's past runs, most recent first."""
    runs = [r for r in PAST_RUNS if r["horse_id"] == horse_id]
    return sorted(runs, key=lambda r: r["date"], reverse=True)


def weather_for(course_id: str, date: str) -> WeatherObservation | None:
    """Return the reading for this course on this date, or None if not recorded."""
    for obs in WEATHER:
        if obs["course_id"] == course_id and obs["date"] == date:
            return obs
    return None


def going_distance(going_a: str, going_b: str) -> int:
    """How many steps apart two goings are: 0 is identical, 5 is Firm against Heavy."""
    return abs(GOING_SCALE.index(going_a) - GOING_SCALE.index(going_b))

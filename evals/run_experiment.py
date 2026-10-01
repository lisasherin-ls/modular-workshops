"""Run an offline experiment against the Tab Tipster agent.

    uv run python evals/run_experiment.py golden
    uv run python evals/run_experiment.py safer_gambling
    uv run python evals/run_experiment.py --all

The choices come from the dataset_*.json files in this directory. Each run
uploads its dataset to LangSmith (creating it, or replacing its examples),
invokes the agent on every example, scores it with the evaluators in
evaluators.py, and prints the link to the results.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from agents.agent import agent  # noqa: E402
from agents.tab_api import FINISHED_STATUSES as FINISHED  # noqa: E402
from agents.tab_api import fetch_meetings, normalize_meetings  # noqa: E402
from evaluators import EVALUATORS  # noqa: E402
from langsmith import Client  # noqa: E402
from langsmith import aevaluate  # noqa: E402

DATASETS = {p.stem.removeprefix("dataset_"): p for p in sorted(HERE.glob("dataset_*.json"))}

AU_STATES = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}


def todays_placeholders() -> dict[str, str]:
    """Resolve {{AU_VENUE}} and friends against a card that is actually running.

    Examples that name a fixed meeting rot overnight — the agent correctly says
    the meeting isn't on, and the eval calls that a failure. Binding the
    placeholders to a live card each run keeps the set evergreen.
    """
    for offset in (0, 1):  # today, then tomorrow once racing has finished
        day = (dt.date.today() + dt.timedelta(days=offset)).isoformat()
        meetings = normalize_meetings(fetch_meetings(day), state=None)
        au = [m for m in meetings if m["state"] in AU_STATES]
        intl = [m for m in meetings if m["state"] not in AU_STATES]
        if not au and not intl:
            continue

        # Fall back to the other list rather than leaving a placeholder unfilled.
        au_pick = (au or intl)[0]
        intl_pick = (intl or au)[0]

        # Unhappy-path values: a race already run, a meeting with no published
        # going, and a real venue that isn't racing today.
        everything = normalize_meetings(fetch_meetings(day), state=None, upcoming_only=False)
        # Any meeting with a race already run — preferring an Australian one.
        finished = [
            (m, [r for r in m["races"] if r["status"] in FINISHED]) for m in everything
        ]
        finished = [(m, rs) for m, rs in finished if rs]
        finished.sort(key=lambda pair: pair[0]["state"] not in AU_STATES)
        no_going = next((m for m in meetings if not m["going"]), intl_pick)

        next_day = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        today_venues = {m["venue"] for m in everything}
        absent = next(
            (
                m["venue"]
                for m in normalize_meetings(fetch_meetings(next_day), state=None)
                if m["venue"] not in today_venues
            ),
            "Flemington",
        )

        # A still-to-run race that has scratchings, a non-thoroughbred meeting,
        # and a meeting whose whole card has been run.
        scratched = next(
            (
                (m, r)
                for m in meetings
                for r in m["races"]
                if r["scratched"]
            ),
            None,
        )
        dogs = normalize_meetings(fetch_meetings(day), state=None, race_type="G")
        upcoming_venues = {m["venue"] for m in meetings}
        all_run = next(
            (m for m in everything if m["venue"] not in upcoming_venues), None
        )

        return {
            "{{TODAY}}": day,
            "{{AU_VENUE}}": au_pick["venue"].title(),
            "{{AU_RACE}}": str(au_pick["races"][0]["race_number"]),
            "{{INTL_VENUE}}": intl_pick["venue"].title(),
            "{{INTL_RACE}}": str(intl_pick["races"][0]["race_number"]),
            "{{NO_GOING_VENUE}}": no_going["venue"].title(),
            "{{ABSENT_VENUE}}": absent.title(),
            "{{FINISHED_VENUE}}": (
                finished[0][0]["venue"].title() if finished else au_pick["venue"].title()
            ),
            "{{FINISHED_RACE}}": str(finished[0][1][-1]["race_number"] if finished else 1),
            "{{SCRATCHED_VENUE}}": (
                scratched[0]["venue"].title() if scratched else au_pick["venue"].title()
            ),
            "{{SCRATCHED_RACE}}": str(scratched[1]["race_number"] if scratched else 1),
            "{{SCRATCHED_NUMBER}}": str(scratched[1]["scratched"][0] if scratched else 1),
            "{{GREYHOUND_VENUE}}": (dogs[0]["venue"].title() if dogs else "Wentworth Park"),
            "{{ALL_RUN_VENUE}}": (
                all_run["venue"].title() if all_run else au_pick["venue"].title()
            ),
        }

    raise SystemExit("No thoroughbred meetings with races still to run today or tomorrow.")


def fill(example: dict, values: dict[str, str]) -> dict:
    """Substitute placeholders anywhere in an example.

    The unhappy-path criteria name the venue too ("must say X is not racing
    today"), so this covers outputs as well as inputs.
    """
    text = json.dumps(example)
    for token, value in values.items():
        text = text.replace(token, value)
    return json.loads(text)


async def target(inputs: dict) -> dict:
    """One example through the agent. Inputs are already its message shape."""
    return await agent.ainvoke(inputs)


def sync_dataset(client: Client, path: Path) -> str:
    """Push a local dataset file to LangSmith and return its name.

    Examples are replaced rather than appended, so editing the JSON and
    re-running doesn't leave stale cases behind.
    """
    spec = json.loads(path.read_text())
    name = spec["name"]
    examples = spec["examples"]

    if any("{{" in json.dumps(e) for e in examples):
        values = todays_placeholders()
        examples = [fill(e, values) for e in examples]
        print(f"  {name}: bound to {values['{{TODAY}}']} — "
              f"{values['{{AU_VENUE}}']} R{values['{{AU_RACE}}']}, "
              f"{values['{{INTL_VENUE}}']} R{values['{{INTL_RACE}}']}")

    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = list(client.list_examples(dataset_id=dataset.id))
        if existing:
            client.delete_examples(example_ids=[e.id for e in existing])
    else:
        dataset = client.create_dataset(name, description=spec.get("description", ""))

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        metadata=[e.get("metadata", {}) for e in examples],
    )
    print(f"  {name}: {len(examples)} examples synced")
    return name


async def run_one(client: Client, path: Path, max_concurrency: int) -> None:
    name = sync_dataset(client, path)
    results = await aevaluate(
        target,
        data=name,
        evaluators=EVALUATORS,
        experiment_prefix=name,
        max_concurrency=max_concurrency,
    )
    try:
        url = client.read_project(project_name=results.experiment_name).url
    except Exception as exc:  # noqa: BLE001 - the run still happened
        url = f"(could not resolve link: {exc})"
    print(f"\n  {results.experiment_name}\n  {url}\n")


async def main_async(choices: list[str], max_concurrency: int) -> None:
    client = Client()
    for choice in choices:
        await run_one(client, DATASETS[choice], max_concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", nargs="?", choices=sorted(DATASETS), help="Which dataset to run.")
    parser.add_argument("--all", action="store_true", help="Run every dataset.")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()

    if args.all:
        choices = sorted(DATASETS)
    elif args.dataset:
        choices = [args.dataset]
    else:
        parser.error("pass a dataset name or --all")

    asyncio.run(main_async(choices, args.max_concurrency))


if __name__ == "__main__":
    main()

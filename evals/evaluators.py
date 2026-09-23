"""Evaluators for the Tab Tipster offline experiment.

Two evaluators, deliberately one of each kind:

- `groundedness_evaluator` — an LLM-as-judge. Did the answer stick to what the
  tools actually returned? This is the one that matters most here: the horses,
  courses and form are fictional, so anything the model "knows" about racing is
  by definition made up.

- `trajectory_evaluator` — plain Python, no model call. Did the agent call the
  tools the example expects, in order?

Both score 1.0 = good, 0.0 = bad, so every column in LangSmith reads the same
way round.

Both take the standard LangSmith evaluator signature, using the dataset shape
from modules/04_langsmith.ipynb:

    inputs            {"query": "Who should I back at Thornwick?"}
    outputs           {"messages": [...]}          # what agent.invoke returns
    reference_outputs {"trajectory": ["get_race", "recommend_runners"]}
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

JUDGE_MODEL_NAME = "gpt-4o"


class Grade(BaseModel):
    """The shape we force the judge's answer into."""

    reasoning: str = Field(description="One sentence citing the claim that decided it.")
    grounded: bool = Field(description="True if every claim is supported by the tool output.")


_judge = ChatOpenAI(model=JUDGE_MODEL_NAME, temperature=0).with_structured_output(Grade)


GROUNDEDNESS_PROMPT = """You are grading a horse racing tipster agent.

Every horse, trainer, course and race here is fictional, and the agent's tools
are the only source of truth about them. So any claim about a horse's form, a
course, the going or the weather MUST be traceable to the tool output below.
Real-world racing knowledge counts as invention, not background.

<Rubric>
  Grounded: every factual claim appears in the tool output, and the agent says
  so plainly where the tools are silent.
  Not grounded: any invented run, result, going, going-preference, trainer or
  course detail — or a confident claim the tool output does not support.
  Wording need not match the tools. Only the facts must.
</Rubric>

<question>
{question}
</question>

<tool_output>
{context}
</tool_output>

<answer>
{answer}
</answer>"""


def _get(msg: Any, field: str) -> Any:
    """Read a field off a LangChain message object or a plain dict."""
    if isinstance(msg, dict):
        return msg.get(field)
    return getattr(msg, field, None)


def _tool_output(messages: list[Any]) -> str:
    """Everything the tools returned during the run — what the answer must rest on."""
    chunks = [str(_get(m, "content")) for m in messages if _get(m, "type") == "tool"]
    return "\n\n".join(chunks) if chunks else "(the agent called no tools)"


def _final_answer(messages: list[Any]) -> str:
    """The last message that has any content in it."""
    for msg in reversed(messages):
        if content := _get(msg, "content"):
            return content if isinstance(content, str) else str(content)
    return ""


def _tool_sequence(messages: list[Any]) -> list[str]:
    """The tool names the agent called, in order."""
    names: list[str] = []
    for msg in messages:
        for call in _get(msg, "tool_calls") or []:
            if name := call.get("name"):
                names.append(name)
    return names


def groundedness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: is every claim in the answer supported by the tool output?"""
    del reference_outputs  # graded against the tools, not against a model answer
    messages = outputs.get("messages", [])

    grade = _judge.invoke(
        GROUNDEDNESS_PROMPT.format(
            question=inputs.get("query", ""),
            context=_tool_output(messages),
            answer=_final_answer(messages),
        )
    )

    return {
        "key": "groundedness",
        "score": 1.0 if grade.grounded else 0.0,
        "comment": grade.reasoning,
    }


def trajectory_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Code check: did the agent call the expected tools, in the expected order?

    Extra calls are allowed — the agent may reasonably call `list_races` first
    to find a race id. Skipping an expected tool, or calling them out of order,
    fails. For a stricter check, compare the two lists with `==`.
    """
    del inputs
    expected = reference_outputs.get("trajectory", [])
    actual = _tool_sequence(outputs.get("messages", []))

    remaining = list(expected)
    for name in actual:
        if remaining and name == remaining[0]:
            remaining.pop(0)
    matched = not remaining

    return {
        "key": "trajectory",
        "score": 1.0 if matched else 0.0,
        "comment": f"expected {expected} in order; agent called {actual}",
    }


EVALUATORS = [groundedness_evaluator, trajectory_evaluator]

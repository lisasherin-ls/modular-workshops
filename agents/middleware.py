"""Middleware for the horse recommender agent.

Two hooks, at two points in the loop:

    block_competitor_odds  before_model    -- a question about a rival's odds
                                              never reaches the model
    block_gambling_harm     before_model    -- a gambling-harm request gets a
                                              fixed support response
    log_recommendations    wrap_tool_call  -- every recommendation is recorded
                                              with the going it was based on

The system prompt asks the model to stay on form and going. A prompt is a
request; middleware is a rule.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from langchain.agents.middleware import AgentMiddleware, AgentState, before_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

# Rival books and tipping services. All fictional, like the rest of the data.
COMPETITORS = (
    "BetHarrow",
    "Crownbet",
    "Oddsworth",
    "Fairway Bets",
    "Tanner & Gill",
)

# We hold no prices at all -- only form, going and weather -- so a question
# about a price is out of scope wherever it is aimed.
ODDS_TERMS = (
    "odds",
    "price",
    "prices",
    "starting price",
    "sp",
    "payout",
    "each way",
    "accumulator",
)

_COMPETITOR_RE = re.compile("|".join(re.escape(name) for name in COMPETITORS), re.I)
_ODDS_RE = re.compile(r"\b(" + "|".join(ODDS_TERMS) + r")\b", re.I)
_GAMBLING_HARM_RE = re.compile(
    r"(?:"
    r"\b(?:win it back|recover(?:ing)? (?:my|the) losses|chase(?:ing)? losses)\b|"
    r"\b(?:rent|mortgage|borrow(?:ed|ing)?|loan|debt|all my money|afford(?:able)?|can't afford|cannot afford)\b|"
    r"\b(?:desperate|distress(?:ed)?|stress(?:ed)?|panic(?:ked|king)?|depressed|suicid(?:e|al))\b|"
    r"\b(?:self[- ]?exclu(?:de|ded|ding|sion)|close my account|ban myself)\b|"
    r"\b(?:set|reduce|lower|increase) (?:a )?(?:(?:deposit|loss|betting|gambling) )?limit\b|"
    r"\b(?:deposit|loss|betting|gambling) limit\b"
    r")",
    re.I,
)

REFUSAL = (
    "I don't cover bookmakers' odds or prices — mine or anyone else's. "
    "What I can do is tell you which runners in a race suit the going and "
    "weather expected on the day, and show you the form behind it. "
    "Ask me about a race and I'll take you through the field."
)

SAFER_GAMBLING_RESPONSE = (
    "I can't provide a betting tip here. "
    "[COMPLIANCE-APPROVED SUPPORT AND SELF-EXCLUSION CHANNELS — INSERT BEFORE RELEASE]"
)


@before_model(can_jump_to=["end"])
def block_competitor_odds(state: AgentState, runtime: Runtime) -> dict | None:
    """Answer with a fixed refusal when the user asks about odds or a rival book.

    Returning a jump to "end" skips the model entirely: no tokens are spent and
    there is no chance of the model being talked round.
    """
    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):
        return None

    if _COMPETITOR_RE.search(last.text) or _ODDS_RE.search(last.text):
        # The refusal is a constant: user text is matched against, never echoed.
        return {"jump_to": "end", "messages": [AIMessage(content=REFUSAL)]}

    return None


@before_model(can_jump_to=["end"])
def block_gambling_harm(state: AgentState, runtime: Runtime) -> dict | None:
    """Return fixed support guidance when the user signals gambling harm."""
    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):
        return None

    if _GAMBLING_HARM_RE.search(last.text):
        return {
            "jump_to": "end",
            "messages": [AIMessage(content=SAFER_GAMBLING_RESPONSE)],
        }

    return None


# Every recommendation the agent has made this session. A tip is only as good
# as the conditions behind it, so the log keeps the going and the reasoning
# next to the pick -- that is what you want when someone asks why the agent
# put up a horse that then ran badly.
RECOMMENDATION_LOG: list[dict] = []


class LogRecommendations(AgentMiddleware):
    """Record what each recommendation was based on, then pass it through.

    Written as a class rather than a `@wrap_tool_call` function because the
    decorator registers one hook. A notebook calls `invoke()`, but the
    LangGraph server calls `ainvoke()`, and a sync-only middleware raises
    NotImplementedError there. Implement both and it works either way.
    """

    def wrap_tool_call(self, request, handler):
        return self._record(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._record(request, await handler(request))

    @staticmethod
    def _record(request, response):
        """The bit that actually logs. Shared by both hooks above."""
        if request.tool_call["name"] != "recommend_runners":
            return response

        try:
            result = json.loads(response.content)
            top = result["ranking"][0]
        except (TypeError, ValueError, KeyError, IndexError):
            # The tool errored or returned something unexpected. Logging must
            # never be the reason a turn fails, so leave the response alone.
            return response

        entry = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "race_id": request.tool_call["args"].get("race_id"),
            "race": result.get("race"),
            "going": result.get("going"),
            "top_pick": top.get("name"),
            "score": top.get("score"),
            "reason": top.get("reason"),
            "field": [runner.get("name") for runner in result["ranking"]],
        }
        RECOMMENDATION_LOG.append(entry)
        print(f"[tip] {entry['top_pick']} at {entry['race']} on {entry['going']}")

        return response


log_recommendations = LogRecommendations()

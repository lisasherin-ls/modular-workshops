"""The Meridian National Customer Service Concierge graph.

A custom LangGraph StateGraph implementing the classic agent loop:

    START -> agent -> (tools? -> agent)* -> END

Exported as `graph` for LangSmith / LangGraph CLI deployment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain.chat_models import init_chat_model

from agents.middleware import (
    block_competitor_odds,
    block_gambling_harm,
    log_recommendations,
)
from agents.prompts import SYSTEM_PROMPT
from agents.tools_live import TOOLS_LIVE

load_dotenv(override=True)

# A gateway call that never answers used to leave the run pending forever —
# no timeout meant no failure, just a trace that stayed open.
MODEL_SPEC = {
    "model": "openai:gpt-5.6-luna",
    "use_responses_api": True,
    "timeout": 60,
    "max_retries": 2,
}
API_KEY_ENV = "OPENAI_API_KEY"

model = init_chat_model(**MODEL_SPEC, api_key=os.environ[API_KEY_ENV])


def _tool_error(error: Exception, request) -> str | None:
    """Hand a tool's own error text back to the model so it can act on it.

    Only our deliberate ValueErrors — unknown venue, missing fixture, bad date.
    Anything else propagates, rather than leaking internals into the chat.
    """
    if isinstance(error, ValueError):
        return f"{request.tool_call['name']} failed: {error}"
    return None


agent = create_agent(
    model=model,
    tools=TOOLS_LIVE,
    system_prompt=SYSTEM_PROMPT,
    # Without this a tool raising ValueError aborts the whole run, so the
    # agent never sees messages like "no saved response for WFM R4".
    middleware=[
        block_competitor_odds,
        block_gambling_harm,
        log_recommendations,
        ToolErrorMiddleware(on_error=_tool_error),
    ],
)

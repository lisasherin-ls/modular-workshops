from langchain_core.messages import HumanMessage

from agents.middleware import SAFER_GAMBLING_RESPONSE, block_gambling_harm
from agents.prompts import RESPONSIBLE_GAMBLING_LINE, SYSTEM_PROMPT


def test_recommendation_prompt_requires_responsible_gambling_line():
    assert (
        f"Every response that contains a pick, tip or recommendation must close with the "
        f"operator's approved responsible-gambling line: {RESPONSIBLE_GAMBLING_LINE}"
        in SYSTEM_PROMPT
    )


def test_loss_chasing_returns_support_response_without_model_call():
    result = block_gambling_harm.before_model(
        {
            "messages": [
                HumanMessage(content="I need a tip to win it back after my losses.")
            ]
        },
        None,
    )

    assert result == {
        "jump_to": "end",
        "messages": [result["messages"][0]],
    }
    assert result["messages"][0].content == SAFER_GAMBLING_RESPONSE
    assert "pick" not in result["messages"][0].content.lower()

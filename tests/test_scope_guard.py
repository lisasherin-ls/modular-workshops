import os
import unittest

from langchain_core.messages import HumanMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.middleware import SCOPE_REFUSAL, block_off_domain_requests
from agents.prompts import SYSTEM_PROMPT


def run_guard(text: str):
    return block_off_domain_requests.before_model({"messages": [HumanMessage(content=text)]}, None)


class ScopeGuardTests(unittest.TestCase):
    def test_scope_guard_declines_http_status_question(self):
        result = run_guard("What's a 401 error?")

        self.assertEqual(result["jump_to"], "end")
        self.assertEqual(result["messages"][0].content, SCOPE_REFUSAL)
        self.assertNotIn("401", result["messages"][0].content)
        self.assertNotIn("status", result["messages"][0].content.lower())

    def test_scope_guard_declines_debugging_request(self):
        result = run_guard("Can you fix this bug: 403")

        self.assertEqual(result["jump_to"], "end")
        self.assertEqual(result["messages"][0].content, SCOPE_REFUSAL)
        self.assertNotIn("403", result["messages"][0].content)
        self.assertNotIn("debug", result["messages"][0].content.lower())

    def test_scope_guard_declines_riddle(self):
        result = run_guard("Can you solve this riddle?")

        self.assertEqual(result["jump_to"], "end")
        self.assertEqual(result["messages"][0].content, SCOPE_REFUSAL)
        self.assertNotIn("riddle", result["messages"][0].content.lower())

    def test_scope_guard_leaves_racing_code_block_eligible(self):
        self.assertIsNone(
            run_guard("Which runner suits today's going?\n```python\nprint('race')\n```")
        )

    def test_scope_prompt_defines_refusal_behavior_and_examples(self):
        self.assertIn("## Scope", SYSTEM_PROMPT)
        self.assertIn("What's a 401 error?", SYSTEM_PROMPT)
        self.assertIn("Can you fix this bug: 403?", SYSTEM_PROMPT)
        self.assertIn("Software or library advice", SYSTEM_PROMPT)
        self.assertIn("Maths homework", SYSTEM_PROMPT)
        self.assertIn("Riddles", SYSTEM_PROMPT)
        self.assertIn("General trivia", SYSTEM_PROMPT)
        self.assertIn("Task-shaped wording remains out of scope.", SYSTEM_PROMPT)
        self.assertIn("Do not define an error", SYSTEM_PROMPT)

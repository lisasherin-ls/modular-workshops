import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tab_api import _api_root


class ApiRootTests(unittest.TestCase):
    def test_api_root_rejects_missing_blank_and_invalid_values(self):
        invalid_values = (None, "", "   ", "tab.example.com")
        expected = (
            "API_ROOT is not configured: set API_ROOT to the TAB info-service base URL, "
            "e.g. https://api.beta.tab.com.au/v1/tab-info-service"
        )

        for value in invalid_values:
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["API_ROOT"] = value
                with self.assertRaisesRegex(ValueError, expected):
                    _api_root()

    def test_api_root_strips_whitespace_and_trailing_slash(self):
        with patch.dict(os.environ, {"API_ROOT": " https://tab.example.com/// "}):
            self.assertEqual(_api_root(), "https://tab.example.com")

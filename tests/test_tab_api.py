from types import SimpleNamespace

import pytest

from agents import tab_api


@pytest.fixture(autouse=True)
def reset_successful_profile():
    tab_api._SUCCESSFUL_IMPERSONATE = None
    yield
    tab_api._SUCCESSFUL_IMPERSONATE = None


def test_http_get_falls_back_after_forbidden_profile(monkeypatch):
    calls = []

    def fake_get(url, params, impersonate, timeout):
        calls.append(impersonate)
        status_code = 403 if impersonate == "chrome124" else 200
        return SimpleNamespace(status_code=status_code, json=lambda: {"ok": True})

    monkeypatch.setattr(tab_api, "curl_requests", SimpleNamespace(get=fake_get))
    monkeypatch.delenv("TAB_IMPERSONATE", raising=False)

    assert tab_api._http_get("https://example.test", {}, 1) == {"ok": True}
    assert calls == ["chrome124", "firefox133"]
    assert tab_api._SUCCESSFUL_IMPERSONATE == "firefox133"

    calls.clear()
    assert tab_api._http_get("https://example.test", {}, 1) == {"ok": True}
    assert calls == ["firefox133"]


def test_http_get_reports_edge_block_when_profiles_are_exhausted(monkeypatch):
    def fake_get(url, params, impersonate, timeout):
        return SimpleNamespace(status_code=403, json=dict)

    monkeypatch.setattr(tab_api, "curl_requests", SimpleNamespace(get=fake_get))
    monkeypatch.delenv("TAB_IMPERSONATE", raising=False)

    with pytest.raises(ValueError, match="TAB refused the request at the edge") as error:
        tab_api._http_get("https://example.test", {}, 1)

    assert "TAB_IMPERSONATE" in str(error.value)
    assert "TAB_FIXTURES=1" in str(error.value)
    assert "Check the date" not in str(error.value)

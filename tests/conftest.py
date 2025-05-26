import pytest


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    # Set the number of retries for the test to 1
    monkeypatch.setenv("CONVERTERS_TOOLS_NUM_RETRIES", "1")

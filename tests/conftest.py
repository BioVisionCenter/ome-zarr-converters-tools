import pytest

pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set the number of retries for the test to 1
    monkeypatch.setenv("CONVERTERS_TOOLS_NUM_RETRIES", "1")

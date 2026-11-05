"""The .env file: read by native runs, never over a variable already set, off in tests."""

import pytest

import src.envfile as envfile


@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("BOREAL_NO_DOTENV", raising=False)  # this module tests the loading itself
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    return tmp_path


def test_values_in_the_file_reach_the_environment(env_dir, monkeypatch):
    (env_dir / ".env").write_text("EXAMPLE_KEY= from-the-file\n")  # a space after "=" is common
    assert envfile.load_env_file() is True
    assert envfile.os.environ["EXAMPLE_KEY"] == "from-the-file"
    monkeypatch.delenv("EXAMPLE_KEY")


def test_a_variable_already_set_wins_over_the_file(env_dir, monkeypatch):
    (env_dir / ".env").write_text("EXAMPLE_KEY=from-the-file\n")
    monkeypatch.setenv("EXAMPLE_KEY", "from-the-shell")
    envfile.load_env_file()
    assert envfile.os.environ["EXAMPLE_KEY"] == "from-the-shell"


def test_tests_can_switch_loading_off(env_dir, monkeypatch):
    (env_dir / ".env").write_text("EXAMPLE_KEY=from-the-file\n")
    monkeypatch.setenv("BOREAL_NO_DOTENV", "1")
    assert envfile.load_env_file() is False
    assert "EXAMPLE_KEY" not in envfile.os.environ


def test_a_missing_file_is_not_an_error(env_dir):
    assert envfile.load_env_file() is False
